import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

@st.cache_data(show_spinner=False, ttl=1800)
def get_data(ticker, start_date, end_date):
    """
    FinanceDataReader(FDR)를 기반으로 주가 데이터를 가져옵니다.
    한국 주식(국장), 미국 주식(미장), 암호화폐 등을 모두 지원합니다.
    """
    if not ticker:
        return pd.DataFrame()

    # 1. 티커 전처리 (FDR 호환용)
    fdr_code = ticker
    is_korean = False
    
    # 한국 주식 (.KS, .KQ 제거 -> 숫자 코드만 남김)
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        fdr_code = ticker.split('.')[0]
        is_korean = True
    
    # 암호화폐 (BTC-USD 등은 그대로 유지, 혹은 FDR 포맷에 맞게 수정 가능)
    # FDR은 암호화폐 티커로 'BTC/KRW', 'ETH/USD' 등을 씁니다.
    # 만약 사용자가 'BTC-USD' (Yahoo식)로 입력했다면 변환 필요할 수 있음.
    
    try:
        # 2. FDR로 데이터 조회
        # FDR은 start, end가 문자열이나 datetime 모두 가능
        df = fdr.DataReader(fdr_code, start_date, end_date)
        
        # 데이터가 없는 경우 (상장폐지, 티커 오류 등)
        if df is None or df.empty:
            return pd.DataFrame()

        # 3. 데이터 표준화 (앱 전체에서 통일된 컬럼명 사용)
        # FDR은 인덱스가 Date인 경우가 많음 -> 컬럼으로 뺌
        df = df.reset_index()
        
        # 컬럼 이름이 'Date'인지 'index'인지 확인 후 'Date'로 통일
        if 'Date' not in df.columns:
            if 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            else:
                # 인덱스 이름조차 없으면 첫 번째 컬럼을 날짜로 간주
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        # 날짜 형식 통일 (datetime64)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 4. 필수 컬럼 존재 여부 확인 및 이름 변경
        # FDR은 보통 'Open', 'High', 'Low', 'Close', 'Volume'을 반환함
        # 하지만 일부 소스(FRED 등)는 다를 수 있음. 주식 기준으론 대부분 맞음.
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                # 만약 Open/High/Low가 없고 Close만 있다면(종가 데이터만 있는 경우) 복사해서 채움
                if 'Close' in df.columns:
                    df[col] = df['Close']
        
        # Change(등락률) 컬럼은 있으면 좋고 없으면 말고
        
        # 5. 정렬 (날짜 오름차순)
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        # FDR 실패 시 yfinance로 재시도 (백업)
        try:
            df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df_yf.empty: return pd.DataFrame()
            df_yf = df_yf.reset_index()
            return df_yf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except:
            print(f"데이터 로드 실패 ({ticker}): {e}")
            return pd.DataFrame()

import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

# --- 1. 주가/시세 데이터 가져오기 (FDR + Yfinance 하이브리드) ---
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
    
    try:
        # 2. FDR로 데이터 조회
        # FDR은 start, end가 문자열이나 datetime 모두 가능
        df = fdr.DataReader(fdr_code, start_date, end_date)
        
        # 데이터가 없는 경우
        if df is None or df.empty:
            return pd.DataFrame()

        # 3. 데이터 표준화 (앱 전체에서 통일된 컬럼명 사용)
        df = df.reset_index()
        
        # 날짜 컬럼 이름 통일 ('Date' 또는 'index' -> 'Date')
        if 'Date' not in df.columns:
            if 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            else:
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        # 날짜 형식 통일
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 4. 필수 컬럼 확보 (Open, High, Low, Close, Volume)
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                # 종가만 있는 경우 다른 컬럼도 종가로 채움
                if 'Close' in df.columns:
                    df[col] = df['Close']
        
        # 5. 정렬
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 필요한 컬럼만 리턴 (Volume이 없으면 0으로 채움)
        if 'Volume' not in df.columns:
            df['Volume'] = 0
            
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        # FDR 실패 시 yfinance로 재시도 (백업)
        try:
            df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df_yf.empty: return pd.DataFrame()
            df_yf = df_yf.reset_index()
            return df_yf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except:
            return pd.DataFrame()

# --- 2. [복구됨] 기업 기본 정보 가져오기 ---
@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    """
    yfinance를 통해 기업의 기본 정보(이름, 섹터, 시총 등)를 가져옵니다.
    """
    try:
        # 한국 주식도 yfinance info는 꽤 잘 나옵니다 (상세 재무 제외)
        t = yf.Ticker(ticker)
        info = t.info
        
        return {
            "name": info.get("longName", ticker),
            "symbol": info.get("symbol", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "summary": info.get("longBusinessSummary", "정보 없음"),
            "website": info.get("website", ""),
            "forwardPE": info.get("forwardPE", 0),
            "trailingPE": info.get("trailingPE", 0),
            "dividendYield": info.get("dividendYield", 0),
        }
    except Exception as e:
        return None

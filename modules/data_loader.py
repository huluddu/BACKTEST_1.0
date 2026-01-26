import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import streamlit as st

def get_data(ticker, start_date, end_date):
    ticker = ticker.upper()
    df = pd.DataFrame()

    try:
        # 1. 국장 지수 및 특정 티커는 FDR 우선
        fdr_priority = ["VKOSPI", "KS11", "KQ11", "USD/KRW", "VIX", "RVX"]
        
        if ticker in fdr_priority:
            df = fdr.DataReader(ticker, start_date, end_date)
        else:
            # 2. 그 외(미장 등)는 yfinance 시도
            # 최근 yfinance는 '^' 유무에 예민하므로 지수는 붙여줌
            yf_ticker = ticker
            if ticker in ["GSPC", "IXIC", "DJI", "N225"]:
                yf_ticker = f"^{ticker}"
            
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

        # 3. yfinance 실패 시 FDR로 최종 백업
        if df.empty:
            df = fdr.DataReader(ticker, start_date, end_date)

    except Exception as e:
        st.error(f"데이터 로드 중 치명적 에러 ({ticker}): {e}")
        return pd.DataFrame()

    # --- 여기서부터 KeyError: 'Date'를 잡는 핵심 로직 ---
    
    if df.empty:
        return pd.DataFrame()

    # A. yfinance 멀티 인덱스 컬럼 해결 (최근 업데이트 이슈)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # B. 인덱스(날짜)를 컬럼으로 빼내기
    df = df.reset_index()

    # C. 날짜 컬럼 이름을 무조건 'Date'로 강제 통일
    # 'index', 'Date', 'datetime', 'Date' 등 어떤 이름으로 오든 첫 번째 컬럼을 Date로 간주
    first_col = df.columns[0]
    df.rename(columns={first_col: 'Date'}, inplace=True)

    # D. 'Date' 컬럼을 실제 datetime 형식으로 변환 (정렬 및 계산용)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

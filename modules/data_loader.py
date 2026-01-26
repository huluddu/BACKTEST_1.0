import pandas as pd
import yfinance as yf
from pykrx import stock
import streamlit as st

def get_data(ticker, start_date, end_date):
    # 1. 입력값 정규화 (공백 제거 및 대문자)
    ticker = ticker.upper().strip()
    df = pd.DataFrame()

    # pykrx 날짜 형식 (YYYYMMDD)
    s_dt = pd.to_datetime(start_date).strftime('%Y%m%d')
    e_dt = pd.to_datetime(end_date).strftime('%Y%m%d')

    try:
        # 2. VKOSPI는 무조건 pykrx (야후 경로 원천 차단)
        if "VKOSPI" in ticker:
            # 코스피 200 변동성지수 코드 1021
            df = stock.get_index_ohlcv_by_date(s_dt, e_dt, "1021")
            
        # 3. 기타 국장 지수
        elif ticker in ["KS11", "KQ11", "KOSPI", "KOSDAQ"]:
            code_map = {"KS11": "1001", "KOSPI": "1001", "KQ11": "2001", "KOSDAQ": "2001"}
            df = stock.get_index_ohlcv_by_date(s_dt, e_dt, code_map[ticker])

        # 4. 그 외 미장 티커 (TQQQ, VIX, SPY 등)
        else:
            yf_ticker = ticker
            if ticker == "VIX": yf_ticker = "^VIX"
            elif ticker == "GSPC": yf_ticker = "^GSPC"
            
            # yfinance 직접 호출 (fdr.DataReader 절대 사용 금지)
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

    except Exception as e:
        st.error(f"데이터 로딩 에러: {e}")
        return pd.DataFrame()

    # --- 후처리 (KeyError 방지) ---
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # 첫 컬럼을 Date로 강제 통일 (pykrx: '날짜', yf: 'Date')
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # yfinance 멀티인덱스 컬럼 정리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

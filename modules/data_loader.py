import pandas as pd
import yfinance as yf
from pykrx import stock  # 국장 지수용 (야후 안 거침)
import streamlit as st

def get_data(ticker, start_date, end_date):
    ticker = ticker.upper().strip()
    df = pd.DataFrame()

    # pykrx용 날짜 형식 (YYYYMMDD)
    s_dt = pd.to_datetime(start_date).strftime('%Y%m%d')
    e_dt = pd.to_datetime(end_date).strftime('%Y%m%d')

    try:
        # 1. VKOSPI는 무조건 pykrx로 (야후 차단 피하기)
        if ticker == "VKOSPI":
            # 1021 = 코스피 200 변동성지수 고유 코드
            df = stock.get_index_ohlcv_by_date(s_dt, e_dt, "1021")
            
        # 2. 기타 국장 지수
        elif ticker in ["KS11", "KQ11", "KOSPI", "KOSDAQ"]:
            code_map = {"KS11": "1001", "KOSPI": "1001", "KQ11": "2001", "KOSDAQ": "2001"}
            df = stock.get_index_ohlcv_by_date(s_dt, e_dt, code_map[ticker])

        # 3. 미장 티커 (TQQQ, VIX, SPY 등)는 yfinance 전담
        else:
            yf_ticker = ticker
            if ticker == "VIX": yf_ticker = "^VIX"
            if ticker == "GSPC": yf_ticker = "^GSPC"
            
            # progress=False로 로그 지저분해지는 것 방지
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

    except Exception as e:
        st.error(f"[{ticker}] 로딩 중 에러 발생: {e}")
        return pd.DataFrame()

    # --- 데이터가 없는 경우 처리 ---
    if df is None or df.empty:
        st.warning(f"[{ticker}] 데이터를 찾을 수 없습니다. (기간: {start_date} ~ {end_date})")
        return pd.DataFrame()

    # --- 컬럼명 통일 (KeyError: 'Date' 해결) ---
    df = df.reset_index()
    # 첫 번째 컬럼이 무엇이든 'Date'로 강제 변경 (pykrx는 '날짜', yf는 'Date')
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # yfinance 멀티인덱스 컬럼 대응
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

import pandas as pd
import yfinance as yf
from pykrx import stock  # 국장 지수 전문
import streamlit as st

def get_data(ticker, start_date, end_date):
    # 입력값 정규화
    ticker = ticker.upper().strip()
    df = pd.DataFrame()

    # 날짜 형식 변환 (pykrx 전용)
    s_dt = pd.to_datetime(start_date).strftime('%Y%m%d')
    e_dt = pd.to_datetime(end_date).strftime('%Y%m%d')

    try:
        # [수정] 14라인 근처에서 fdr을 완전히 제거함
        if "VKOSPI" in ticker:
            # 1021 = 코스피 200 변동성지수
            df = stock.get_index_ohlcv_by_date(s_dt, e_dt, "1021")
            
        elif ticker in ["KS11", "KQ11", "KOSPI", "KOSDAQ"]:
            code_map = {"KS11": "1001", "KOSPI": "1001", "KQ11": "2001", "KOSDAQ": "2001"}
            df = stock.get_index_ohlcv_by_date(s_dt, e_dt, code_map[ticker])

        else:
            # 미장 및 기타 지수는 yfinance 전담
            yf_ticker = ticker
            if ticker == "VIX": yf_ticker = "^VIX"
            elif ticker == "GSPC": yf_ticker = "^GSPC"
            
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 후처리: 어떤 이름으로 오든 첫 컬럼을 'Date'로 강제 고정
    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # yfinance 멀티인덱스 대응
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

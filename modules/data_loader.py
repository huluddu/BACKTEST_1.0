import yfinance as yf
import pandas as pd
from pykrx import stock  # 국장 전용 구원투수
import streamlit as st
from datetime import datetime

def get_data(ticker, start_date, end_date):
    ticker = ticker.upper()
    df = pd.DataFrame()

    # 날짜 형식 변환 (pykrx는 YYYYMMDD 형식을 좋아합니다)
    s_date = pd.to_datetime(start_date).strftime('%Y%m%d')
    e_date = pd.to_datetime(end_date).strftime('%Y%m%d')

    try:
        # 1. VKOSPI 전용 로직 (pykrx 사용)
        if ticker == "VKOSPI":
            # 1021은 코스피 200 변동성지수(VKOSPI)의 고유 코드입니다.
            df = stock.get_index_ohlcv_by_date(s_date, e_date, "1021")
            
        # 2. 기타 국장 지수 (KOSPI 등) 필요 시
        elif ticker in ["KS11", "KOSPI"]:
            df = stock.get_index_ohlcv_by_date(s_date, e_date, "1001")

        # 3. 그 외 모든 미장 티커 (TQQQ, VIX, SPY 등)
        else:
            yf_ticker = ticker
            if ticker == "VIX": yf_ticker = "^VIX"
            if ticker == "GSPC": yf_ticker = "^GSPC"
            
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

    except Exception as e:
        st.error(f"데이터 로딩 실패 ({ticker}): {e}")
        return pd.DataFrame()

    # --- 에러 방지용 데이터 후처리 ---
    if df.empty:
        st.warning(f"[{ticker}] 데이터가 비어 있습니다. 기간을 확인하세요.")
        return pd.DataFrame()

    # yfinance 멀티인덱스 해제
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 인덱스(날짜)를 컬럼으로 빼내고 이름을 'Date'로 통일
    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    
    # Date 컬럼을 확실하게 datetime 객체로 변환
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

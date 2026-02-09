import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

# [핵심] 에러 방지용 빈 껍데기 데이터프레임 정의
EMPTY_DF = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

@st.cache_data(show_spinner=False, ttl=600)
def get_data(ticker, start_date, end_date):
    if not ticker:
        return EMPTY_DF

    ticker = ticker.strip().upper()
    df = pd.DataFrame()

    # 1. FinanceDataReader 시도 (국장/미장 통합)
    try:
        if ticker.isdigit():
            df = fdr.DataReader(ticker, start_date, end_date)
        else:
            df = fdr.DataReader(ticker, start_date, end_date)
            
        if not df.empty:
            df = df.reset_index()
            return _standardize_df(df)
    except:
        pass

    # 2. 실패 시 yfinance 백업 시도
    try:
        yf_code = f"{ticker}.KS" if ticker.isdigit() else ticker
        df = yf.download(yf_code, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            return _standardize_df(df)
    except:
        pass

    # [중요] 모든 시도 실패 시, 그냥 빈 DF가 아니라 '형식 갖춘 빈 DF' 반환
    return EMPTY_DF

def _standardize_df(df):
    """컬럼 이름을 표준 포맷으로 통일하고, 실패 시 빈 껍데기 반환"""
    try:
        # 날짜 컬럼 통일
        col_map = {c.lower(): c for c in df.columns}
        if 'date' in col_map: 
            df.rename(columns={col_map['date']: 'Date'}, inplace=True)
        elif 'index' in df.columns: 
            df.rename(columns={'index': 'Date'}, inplace=True)
        else: 
            # 인덱스가 날짜인 경우
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            else:
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        # 필수 컬럼 확보 (Open, High, Low, Close)
        required = ['Open', 'High', 'Low', 'Close']
        for req in required:
            for c in df.columns:
                if c.lower() == req.lower(): 
                    df.rename(columns={c: req}, inplace=True)
                    break
        
        # 없는 컬럼은 Close로 채움 (에러 방지)
        if 'Close' in df.columns:
            for req in required:
                if req not in df.columns: df[req] = df['Close']
        else:
            # Close 조차 없으면 빈 껍데기 리턴
            return EMPTY_DF
            
        if 'Volume' not in df.columns: df['Volume'] = 0
        
        # 최종 포맷팅
        df['Date'] = pd.to_datetime(df['Date'])
        # 날짜가 이상한 데이터 필터링
        df = df.dropna(subset=['Date'])
        
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception:
        return EMPTY_DF

@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    default = {
        "Name": ticker, "Symbol": ticker, "Sector": "-", 
        "MarketCap": 0, "Beta": 0.0, "PER": 0, "PBR": 0, "ROE": 0, 
        "NetIncome": 0, "Description": ""
    }
    try:
        target = f"{ticker}.KS" if ticker.isdigit() else ticker
        info = yf.Ticker(target).info
        if not info: return default
        
        return {
            "Name": info.get("longName", ticker),
            "Symbol": info.get("symbol", ticker),
            "Sector": info.get("sector", "N/A"),
            "MarketCap": info.get("marketCap", 0),
            "Beta": info.get("beta", 0.0),
            "PER": info.get("trailingPE", 0),
            "PBR": info.get("priceToBook", 0),
            "ROE": info.get("returnOnEquity", 0),
            "NetIncome": info.get("netIncomeToCommon", 0),
            "Description": info.get("longBusinessSummary", "")
        }
    except:
        return default

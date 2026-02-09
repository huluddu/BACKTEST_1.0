import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

# [중요] pykrx import 라인을 아예 없앴습니다.

@st.cache_data(show_spinner=False, ttl=600)
def get_data(ticker, start_date, end_date):
    if not ticker: return pd.DataFrame()
    ticker = ticker.strip().upper()

    # 1. FDR (국장/미장 통합) 시도
    try:
        # 국장 (숫자만 있는 경우)
        if ticker.isdigit():
            df = fdr.DataReader(ticker, start_date, end_date)
        else:
            # 미장
            df = fdr.DataReader(ticker, start_date, end_date)
            
        if not df.empty:
            df = df.reset_index()
            # 컬럼 이름 표준화
            return _standardize_df(df)
    except:
        pass

    # 2. FDR 실패 시 yfinance 백업 시도
    try:
        # 국장인 경우 .KS 붙여서 시도
        yf_code = f"{ticker}.KS" if ticker.isdigit() else ticker
        df = yf.download(yf_code, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if not df.empty:
            # 멀티인덱스 컬럼 처리 (yfinance 최신버전 이슈)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            return _standardize_df(df)
    except:
        pass

    return pd.DataFrame()

def _standardize_df(df):
    """데이터프레임 컬럼 이름을 표준 포맷(Date, Open, High, Low, Close, Volume)으로 통일"""
    # 날짜 컬럼 통일
    col_map = {c.lower(): c for c in df.columns}
    if 'date' in col_map: df.rename(columns={col_map['date']: 'Date'}, inplace=True)
    elif 'index' in df.columns: df.rename(columns={'index': 'Date'}, inplace=True)
    else: df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # 필수 컬럼 확보
    required = ['Open', 'High', 'Low', 'Close']
    for req in required:
        for c in df.columns:
            if c.lower() == req.lower(): df.rename(columns={c: req}, inplace=True); break
    
    # 없는 컬럼은 Close로 채움
    if 'Close' in df.columns:
        for req in required:
            if req not in df.columns: df[req] = df['Close']
            
    if 'Volume' not in df.columns: df['Volume'] = 0
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return pd.DataFrame()

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

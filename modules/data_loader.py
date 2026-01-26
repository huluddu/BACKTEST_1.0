import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

@st.cache_data(show_spinner=False, ttl=1800)
def get_data(ticker, start_date, end_date):
    if not ticker: return pd.DataFrame()
    fdr_code = ticker.split('.')[0] if ticker.endswith((".KS", ".KQ")) else ticker
    try:
        df = fdr.DataReader(fdr_code, start_date, end_date)
        if df is None or df.empty: return pd.DataFrame()
        df = df.reset_index()
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'} if 'index' in df.columns else {df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in df.columns and 'Close' in df.columns: df[col] = df['Close']
        if 'Volume' not in df.columns: df['Volume'] = 0
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
    except:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True).reset_index()
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    default = {"Name": ticker, "Symbol": ticker, "Sector": "-", "Industry": "-", "MarketCap": 0, "Beta": 0.0, "Summary": "정보 없음", "Website": "", "ForwardPE": 0, "TrailingPE": 0, "DividendYield": 0}
    try:
        info = yf.Ticker(ticker).info
        if not info: return default
        return {
            "Name": info.get("longName", ticker),
            "Symbol": info.get("symbol", ticker),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "MarketCap": info.get("marketCap", 0),
            "Beta": info.get("beta", 0.0),
            "Summary": info.get("longBusinessSummary", "정보 없음"),
            "Website": info.get("website", ""),
            "ForwardPE": info.get("forwardPE", 0),
            "TrailingPE": info.get("trailingPE", 0),
            "DividendYield": info.get("dividendYield", 0)
        }
    except: return default

import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

@st.cache_data(show_spinner=False, ttl=1800)
def get_data(ticker, start_date, end_date):
    # 1. 한국 지수 및 주요 글로벌 지수 (FDR이 더 안정적)
    # VKOSPI, VIX, KOSPI, KOSDAQ, USD/KRW 등
    fdr_tickers = ["VKOSPI", "VIX", "KS11", "KQ11", "USD/KRW"]
    
    if ticker.upper() in fdr_tickers:
        df = fdr.DataReader(ticker, start_date, end_date)
    else:
        # 2. 일반 미국 주식/ETF는 yfinance 시도
        df = yf.download(ticker, start=start_date, end=end_date)
        
    # 3. 데이터가 비어있는지 확인 (KeyError 방지 핵심!)
    if df.empty:
        # yfinance가 실패했다면 마지막 수단으로 FDR 시도
        df = fdr.DataReader(ticker, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"[{ticker}] 데이터를 모든 소스에서 찾을 수 없습니다.")

    # 컬럼명 통일 및 인덱스 초기화
    df = df.reset_index()
    # FDR과 yfinance의 날짜 컬럼명을 'Date'로 통일
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
        
    return df

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


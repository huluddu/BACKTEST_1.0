import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime

# --- 1. 주가/시세 데이터 가져오기 (FDR + Yfinance 하이브리드) ---
@st.cache_data(show_spinner=False, ttl=1800)
def get_data(ticker, start_date, end_date):
    if not ticker:
        return pd.DataFrame()

    fdr_code = ticker
    
    # 한국 주식 (.KS, .KQ 제거)
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        fdr_code = ticker.split('.')[0]
    
    try:
        # FDR로 데이터 조회
        df = fdr.DataReader(fdr_code, start_date, end_date)
        
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        
        # 날짜 컬럼 이름 통일
        if 'Date' not in df.columns:
            if 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            else:
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'])
        
        # 필수 컬럼 확보
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                if 'Close' in df.columns:
                    df[col] = df['Close']
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
            
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        try:
            df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df_yf.empty: return pd.DataFrame()
            df_yf = df_yf.reset_index()
            return df_yf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except:
            return pd.DataFrame()

# --- 2. 기업 기본 정보 가져오기 (키값 대문자 수정) ---
@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    """
    KeyError 방지를 위해 Key 값을 대문자(Name, Sector 등)로 통일했습니다.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # [수정됨] main.py가 찾는 Key 이름(대문자)으로 맞춤
        return {
            "Name": info.get("longName", ticker),       # "name" -> "Name"
            "Symbol": info.get("symbol", ticker),       # "symbol" -> "Symbol"
            "Sector": info.get("sector", "N/A"),        # "sector" -> "Sector"
            "Industry": info.get("industry", "N/A"),    # "industry" -> "Industry"
            "MarketCap": info.get("marketCap", 0),      # "marketCap" -> "MarketCap"
            "Summary": info.get("longBusinessSummary", "정보 없음"), # "summary" -> "Summary"
            "Website": info.get("website", ""),
            "ForwardPE": info.get("forwardPE", 0),
            "TrailingPE": info.get("trailingPE", 0),
            "DividendYield": info.get("dividendYield", 0),
        }
    except Exception as e:
        # 에러 발생 시 빈 딕셔너리라도 리턴해서 멈추지 않게 함
        return {
            "Name": ticker, "Symbol": ticker, "Sector": "-", "Industry": "-", 
            "MarketCap": 0, "Summary": "정보를 불러올 수 없습니다.", "Website": ""
        }

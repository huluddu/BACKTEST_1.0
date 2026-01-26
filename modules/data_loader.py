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

# --- 2. 기업 기본 정보 가져오기 (Beta 추가 및 대문자 Key 통일) ---
@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    """
    Beta, PER, PBR 등 main.py에서 사용하는 모든 Key를 대문자로 리턴합니다.
    """
    # 기본값 설정 (에러 방지용)
    default_info = {
        "Name": ticker, 
        "Symbol": ticker, 
        "Sector": "N/A", 
        "Industry": "N/A", 
        "MarketCap": 0, 
        "Beta": 0.0,            # [중요] Beta 기본값 추가
        "ForwardPE": 0.0,
        "TrailingPE": 0.0,
        "DividendYield": 0.0,
        "Summary": "정보를 불러올 수 없습니다.", 
        "Website": ""
    }

    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # yfinance info가 비어있을 경우 대비
        if not info:
            return default_info

        return {
            "Name": info.get("longName", ticker),
            "Symbol": info.get("symbol", ticker),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "MarketCap": info.get("marketCap", 0),
            
            # [수정] Beta 추가! (없는 경우 0.0)
            "Beta": info.get("beta", 0.0),
            
            "ForwardPE": info.get("forwardPE", 0.0),
            "TrailingPE": info.get("trailingPE", 0.0),
            "DividendYield": info.get("dividendYield", 0.0),
            "Summary": info.get("longBusinessSummary", "정보 없음"),
            "Website": info.get("website", ""),
        }
    except Exception as e:
        # 에러 발생 시에도 멈추지 않도록 기본값 리턴
        return default_info

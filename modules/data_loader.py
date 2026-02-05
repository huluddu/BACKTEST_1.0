import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# 빈 데이터프레임 형틀 (에러 방지용)
EMPTY_DF = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

@st.cache_data(show_spinner=False, ttl=600)
def get_data(ticker, start_date, end_date):
    if not ticker:
        return EMPTY_DF

    ticker = ticker.strip().upper()
    
    # 후보군 생성 (국장 .KS/.KQ 시도)
    candidates = []
    if ticker.isdigit():
        candidates.append(f"{ticker}.KS")
        candidates.append(f"{ticker}.KQ")
    else:
        candidates.append(ticker)

    df = pd.DataFrame()

    for code in candidates:
        try:
            # yfinance 데이터 요청
            temp_df = yf.download(code, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # MultiIndex 컬럼 해결
            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = temp_df.columns.get_level_values(0)

            if not temp_df.empty and len(temp_df) > 5:
                df = temp_df
                break
        except Exception:
            continue

    if df.empty:
        return EMPTY_DF # [핵심] 그냥 빈 DF가 아니라 컬럼이 있는 빈 DF 반환

    # 데이터 표준화
    df = df.reset_index()
    
    # 날짜 컬럼 보정
    col_map = {c.lower(): c for c in df.columns}
    if 'date' in col_map:
        df.rename(columns={col_map['date']: 'Date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # 필수 컬럼 보정
    required = ['Open', 'High', 'Low', 'Close']
    for req in required:
        for col in df.columns:
            if col.lower() == req.lower():
                df.rename(columns={col: req}, inplace=True)
                break
    
    if 'Close' in df.columns:
        for req in required:
            if req not in df.columns:
                df[req] = df['Close']
    
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        # 날짜가 없으면 에러 처리
        if df['Date'].isna().all():
            return EMPTY_DF
            
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return EMPTY_DF

# 기업 정보 (이전과 동일, 유지)
@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    default = {
        "Name": ticker, "Symbol": ticker, "Sector": "N/A", "Industry": "N/A", 
        "MarketCap": 0, "Beta": 0.0, "Summary": "정보 없음", "Website": "", 
        "ForwardPE": 0, "TrailingPE": 0, "DividendYield": 0,
        "PER": 0, "PBR": 0, "ROE": 0, "NetIncome": 0, "Description": ""
    }
    
    target_ticker = ticker
    if ticker.isdigit():
        target_ticker = f"{ticker}.KS"

    try:
        t = yf.Ticker(target_ticker)
        info = t.info
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
            "DividendYield": info.get("dividendYield", 0),
            "PER": info.get("trailingPE", 0),
            "PBR": info.get("priceToBook", 0),
            "ROE": info.get("returnOnEquity", 0),
            "NetIncome": info.get("netIncomeToCommon", 0),
            "Description": info.get("longBusinessSummary", "")
        }
    except:
        return default

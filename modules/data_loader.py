import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

@st.cache_data(show_spinner=False, ttl=600)
def get_data(ticker, start_date, end_date):
    if not ticker:
        return pd.DataFrame()

    ticker = ticker.strip().upper()
    
    # 1. 시도할 티커 후보군 만들기
    candidates = []
    
    # 만약 사용자가 '005930' 처럼 숫자만 넣었다면?
    if ticker.isdigit():
        candidates.append(f"{ticker}.KS")  # 1순위: 코스피 시도
        candidates.append(f"{ticker}.KQ")  # 2순위: 코스닥 시도
    else:
        # 이미 .KS나 .KQ를 붙였거나 미국 주식이면 그대로 사용
        candidates.append(ticker)
        # 혹시 실수로 .KS를 안 붙였을 미국 티커 등을 위해 원본도 유지

    df = pd.DataFrame()

    # 2. 후보군 순회하며 데이터 요청 (야후 파이낸스 강제 사용)
    for code in candidates:
        try:
            # yfinance 다운로드
            temp_df = yf.download(code, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # 멀티인덱스 컬럼 버그 해결 (Price, Ticker) -> Price
            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = temp_df.columns.get_level_values(0)

            # 데이터가 비어있지 않고, 행 개수가 충분하면 성공으로 간주
            if not temp_df.empty and len(temp_df) > 5:
                df = temp_df
                break # 성공했으니 반복문 탈출
        except Exception as e:
            continue

    # 3. 데이터가 여전히 없으면 빈 DF 반환
    if df.empty:
        return pd.DataFrame()

    # 4. 데이터 표준화 (이름 변경 등)
    df = df.reset_index()
    
    # 날짜 컬럼 통일
    col_map = {c.lower(): c for c in df.columns}
    if 'date' in col_map:
        df.rename(columns={col_map['date']: 'Date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # 필수 컬럼 확보
    required = ['Open', 'High', 'Low', 'Close']
    for req in required:
        # 대소문자 매칭 (open -> Open)
        for col in df.columns:
            if col.lower() == req.lower():
                df.rename(columns={col: req}, inplace=True)
                break
    
    # 없는 컬럼은 Close로 채움
    if 'Close' in df.columns:
        for req in required:
            if req not in df.columns:
                df[req] = df['Close']
    
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return pd.DataFrame()

# ---------------------------------------------------------
# 기업 기본 정보 (야후 파이낸스 전용)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    default = {
        "Name": ticker, "Symbol": ticker, "Sector": "-", "Industry": "-", 
        "MarketCap": 0, "Beta": 0.0, "Summary": "정보 없음", "Website": "", 
        "ForwardPE": 0, "TrailingPE": 0, "DividendYield": 0
    }
    
    # 숫자만 들어오면 .KS(코스피)로 가정하고 시도
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
            "DividendYield": info.get("dividendYield", 0)
        }
    except:
        return default

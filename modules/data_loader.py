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
    
    # 1. 시도할 티커 후보군 생성
    candidates = []
    if ticker.isdigit(): # '005930' 입력 시
        candidates.append(f"{ticker}.KS") # 코스피 우선
        candidates.append(f"{ticker}.KQ") # 코스닥 차선
    else:
        candidates.append(ticker) # 'AAPL', 'SOXL' 등

    df = pd.DataFrame()

    # 2. 후보군 순회하며 데이터 요청
    for code in candidates:
        try:
            # [변경점] download 대신 Ticker().history 사용 (더 안정적)
            t = yf.Ticker(code)
            temp_df = t.history(start=start_date, end=end_date, auto_adjust=True)
            
            # 데이터가 비어있으면 다음 후보로(예: .KS 실패 -> .KQ 시도)
            if temp_df.empty:
                continue

            # [중요] yfinance 최신 버전 호환성 (Timezone 제거)
            if temp_df.index.tz is not None:
                temp_df.index = temp_df.index.tz_localize(None)

            # 데이터가 5개 이상 있어야 유효하다고 판단
            if len(temp_df) > 5:
                df = temp_df
                break # 성공했으니 탈출
                
        except Exception as e:
            print(f"Error fetching {code}: {e}")
            continue

    # 3. 모든 시도가 실패했으면 빈 DF 반환
    if df.empty:
        return EMPTY_DF

    # 4. 데이터 표준화 (이름 변경 등)
    # yf.Ticker().history()는 인덱스가 Date입니다.
    df = df.reset_index()
    
    # 컬럼 이름 통일 로직
    col_map = {c.lower(): c for c in df.columns}
    
    # 날짜 컬럼 찾기
    if 'date' in col_map:
        df.rename(columns={col_map['date']: 'Date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # OHLCV 컬럼 확보
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for req in required:
        found = False
        for col in df.columns:
            if col.lower() == req.lower():
                df.rename(columns={col: req}, inplace=True)
                found = True
                break
        # 없는 컬럼은 Close로 채우거나 0 처리
        if not found and 'Close' in df.columns:
             if req == 'Volume': df[req] = 0
             else: df[req] = df['Close']
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        # 최종적으로 필요한 컬럼만 리턴
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Standardization Error: {e}")
        return EMPTY_DF

# -----------------------------------------------------------
# 기업 기본 정보 (이전과 동일, 유지)
# -----------------------------------------------------------
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

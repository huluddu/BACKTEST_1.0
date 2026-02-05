import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime
import time

# -----------------------------------------------------------
# 1. 데이터 가져오기 (FDR 최우선 -> 실패 시 Yfinance 백업)
# -----------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)  # 10분 캐시 (에러가 길게 남지 않도록)
def get_data(ticker, start_date, end_date):
    if not ticker:
        return pd.DataFrame()

    # 입력값 정리
    ticker = ticker.strip().upper()
    
    # 전략별 티커 포맷 설정
    # FDR용: 005930 (숫자만)
    # YF용: 005930.KS (확장자 필요)
    fdr_code = ticker
    yf_code = ticker

    # 한국 주식 티커 처리 로직
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        fdr_code = ticker.split('.')[0] # FDR은 확장자 제거
        yf_code = ticker                # 야후는 확장자 유지
    elif ticker.isdigit(): # 사용자가 '005930' 처럼 숫자만 넣었을 때
        fdr_code = ticker
        yf_code = f"{ticker}.KS" # 일단 코스피로 가정하고 야후 대기

    df = pd.DataFrame()

    # [시도 1] FinanceDataReader (국장 데이터 최강자)
    try:
        # FDR은 에러가 나도 빈 DF를 줄 때가 있어서 try-except 필수
        df = fdr.DataReader(fdr_code, start_date, end_date)
        if not df.empty:
            df = _standardize_df(df)
            if len(df) > 5: # 데이터가 너무 적으면(상장폐지 등) 실패로 간주
                return df
    except Exception:
        pass # FDR 실패 시 조용히 넘어감 (로그 생략)

    # [시도 2] yfinance (미장 최강자 + 국장 백업)
    try:
        # yfinance 최신버전은 가끔 컬럼이 MultiIndex로 오는 버그가 있음 -> auto_adjust=True 권장
        df = yf.download(yf_code, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # MultiIndex 컬럼 평탄화 (Price, Ticker) -> (Price)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if not df.empty and len(df) > 5:
            return _standardize_df(df)
            
        # [시도 2-1] 국장인데 코스피(.KS)로 실패했다면 코스닥(.KQ)으로 재시도
        if ticker.isdigit() and yf_code.endswith(".KS"):
            yf_code_kq = f"{ticker}.KQ"
            df = yf.download(yf_code_kq, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > 5:
                return _standardize_df(df)

    except Exception as e:
        print(f"Data Load Error ({ticker}): {e}")

    return pd.DataFrame() # 최후의 수단: 빈 데이터프레임 반환

def _standardize_df(df):
    """
    어떤 라이브러리에서 가져오든 앱이 이해할 수 있는 표준 포맷으로 변환
    """
    df = df.reset_index()
    
    # 1. 날짜 컬럼 찾아서 'Date'로 통일
    col_map = {col: col.lower() for col in df.columns}
    if 'date' in col_map.values():
        for c in df.columns:
            if c.lower() == 'date':
                df.rename(columns={c: 'Date'}, inplace=True)
                break
    elif 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    else:
        # 인덱스가 날짜일 확률이 높음, 첫 컬럼을 날짜로 간주
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # 2. 필수 컬럼(OHLC) 대소문자 보정 및 확보
    required = ['Open', 'High', 'Low', 'Close']
    new_cols = {}
    for c in df.columns:
        for r in required:
            if c.lower() == r.lower():
                new_cols[c] = r
    df.rename(columns=new_cols, inplace=True)
    
    # 3. 누락된 컬럼은 종가(Close)로 채우기 (데이터 펑크 방지)
    if 'Close' in df.columns:
        for req in required:
            if req not in df.columns:
                df[req] = df['Close']
    
    # 4. Volume 없으면 0으로
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # 5. 날짜 포맷 및 정렬
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return pd.DataFrame()

# -----------------------------------------------------------
# 2. 기업 기본 정보 가져오기 (에러 방지 강화)
# -----------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker):
    # 기본값 (데이터가 없을 때 앱이 멈추지 않게 함)
    default = {
        "Name": ticker, "Symbol": ticker, "Sector": "-", "Industry": "-", 
        "MarketCap": 0, "Beta": 0.0, "Summary": "정보를 불러올 수 없습니다.", "Website": "", 
        "ForwardPE": 0, "TrailingPE": 0, "DividendYield": 0,
        "PER": 0, "PBR": 0, "ROE": 0, "NetIncome": 0, "Description": ""
    }
    
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # info가 비어있으면 기본값 리턴
        if not info or info.get('regularMarketPrice') is None: 
             # 국장의 경우 가끔 info가 늦게 뜰 수 있음, 티커명만이라도 갱신 시도
             return default
        
        return {
            "Name": info.get("longName", info.get("shortName", ticker)),
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

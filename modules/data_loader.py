import streamlit as st
import pandas as pd
import yfinance as yf
from pykrx import stock
import re
import datetime

def _normalize_krx_ticker(t: str) -> str:
    if not isinstance(t, str): t = str(t or "")
    t = t.strip().upper()
    t = re.sub(r"\.(KS|KQ)$", "", t)
    m = re.search(r"(\d{6})", t)
    return m.group(1) if m else ""

@st.cache_data(show_spinner=False, ttl=3600)
def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        t = (ticker or "").strip()
        if not t: return pd.DataFrame()
        is_krx = t.isdigit() or t.lower().endswith(".ks") or t.lower().endswith(".kq")
        if is_krx:
            code = _normalize_krx_ticker(t)
            s, e = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
            df = stock.get_etf_ohlcv_by_date(s, e, code)
            if df is None or df.empty: df = stock.get_market_ohlcv_by_date(s, e, code)
            if not df.empty:
                df = df.reset_index().rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
        else:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if df.empty:
                df = yf.download(t, period="max", progress=False, auto_adjust=False)
                if not df.empty:
                    df = df[df.index <= pd.Timestamp(end_date)]

            if isinstance(df.columns, pd.MultiIndex):
                try: 
                    if t in df.columns.levels[1]: df = df.xs(t, axis=1, level=1)
                    else: df = df.droplevel(1, axis=1)
                except: df = df.droplevel(1, axis=1)
            
            df = df.reset_index()
            if "Datetime" in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = df["Date"].dt.tz_localize(None)

        if df is None or df.empty: return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        cols = ["Open", "High", "Low", "Close"]
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except Exception as e:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

# [추가됨] 기업 기본 정보 가져오기 (이전 대화에서 요청하신 기능)
@st.cache_data(ttl=3600*24)
def get_fundamental_info(ticker):
    info_dict = {"Name": ticker, "Sector": "N/A", "MarketCap": 0, "PER": 0, "PBR": 0, "ROE": 0, "Revenue": 0, "NetIncome": 0, "Beta": 0, "Description": "정보 없음"}
    try:
        yf_ticker = f"{ticker}.KS" if ticker.isdigit() else ticker
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        info_dict["Name"] = info.get("longName", ticker)
        info_dict["Sector"] = info.get("sector", "N/A")
        info_dict["MarketCap"] = info.get("marketCap", 0)
        info_dict["PER"] = info.get("trailingPE", 0)
        info_dict["PBR"] = info.get("priceToBook", 0)
        info_dict["ROE"] = info.get("returnOnEquity", 0)
        info_dict["NetIncome"] = info.get("netIncomeToCommon", 0)
        info_dict["Beta"] = info.get("beta", 0)
        info_dict["Description"] = info.get("longBusinessSummary", "설명 없음")
        return info_dict
    except: return info_dict
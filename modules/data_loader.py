# data_loader.py 상단에서 fdr 임포트를 아예 제거하세요!
import yfinance as yf
from pykrx import stock
import pandas as pd

def get_data(ticker, start_date, end_date):
    t = ticker.upper().strip()
    
    # 1. VKOSPI는 무조건 pykrx (거래소 직통)
    if "VKOSPI" in t:
        s = pd.to_datetime(start_date).strftime('%Y%m%d')
        e = pd.to_datetime(end_date).strftime('%Y%m%d')
        df = stock.get_index_ohlcv_by_date(s, e, "1021")
    
    # 2. 미장 티커는 yfinance
    else:
        yt = f"^{t}" if t in ["VIX", "GSPC"] else t
        df = yf.download(yt, start=start_date, end=end_date, progress=False)

    # 3. 공통 후처리 (Date 컬럼 강제 생성)
    if df.empty: return pd.DataFrame()
    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    return df

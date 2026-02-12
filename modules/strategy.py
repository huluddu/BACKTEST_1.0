import pandas as pd
import numpy as np
import streamlit as st
import random
import datetime
from .data_loader import get_data

# --- 수학 계산 함수들 ---
def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1: return x.astype(float)
    kernel = np.ones(w, dtype=float) / w
    y = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        conv = np.convolve(x, kernel, mode="valid")
        y[w-1:] = conv
    return y

def calculate_bollinger_bands(close_data, period, std_dev_mult):
    period = int(period)
    close_series = pd.Series(close_data)
    ma = close_series.rolling(window=period).mean()
    std = close_series.rolling(window=period).std()
    upper = ma + (std * std_dev_mult)
    lower = ma - (std * std_dev_mult)
    return ma.to_numpy(), upper.to_numpy(), lower.to_numpy()

def calculate_indicators(close_data, rsi_period):
    rsi_period = int(rsi_period)
    df = pd.DataFrame({'close': close_data})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

def calculate_atr(df, period=14):
    if df.empty: return pd.Series(dtype=float)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# --- 데이터 준비 ---
@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, market_ticker, start_date, end_date, ma_pool, market_ma_period=200):
    sig = get_data(signal_ticker, start_date, end_date)
    if sig is None or sig.empty or 'Date' not in sig.columns: return None, None, None, None, None, None
    sig = sig.sort_values("Date")

    trd = get_data(trade_ticker, start_date, end_date)
    if trd is None or trd.empty or 'Date' not in trd.columns: return None, None, None, None, None, None
    trd = trd.sort_values("Date")
    
    # ATR 계산 (매매 데이터 기준)
    trd["ATR"] = calculate_atr(trd, period=14)

    sig = sig.rename(columns={"Close": "Close_sig", "Open":"Open_sig", "High":"High_sig", "Low":"Low_sig"})
    # 필요한 컬럼만 남기기 (에러 방지)
    sig_cols = [c for c in ["Date", "Close_sig", "Open_sig", "High_sig", "Low_sig"] if c in sig.columns]
    sig = sig[sig_cols]
    
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd", "ATR": "ATR"})
    
    base = pd.merge(sig, trd, on="Date", how="inner")
    
    x_mkt, ma_mkt_arr = None, None
    if market_ticker:
        mkt = get_data(market_ticker, start_date, end_date)
        if not mkt.empty and 'Close' in mkt.columns:
            mkt = mkt.sort_values("Date")
            mkt = mkt.rename(columns={"Close": "Close_mkt"})[["Date", "Close_mkt"]]
            base = pd.merge(base, mkt, on="Date", how="inner")
            
    if base.empty: return None, None, None, None, None, None

    base = base.dropna().reset_index(drop=True)
    
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)

    if "Close_mkt" in base.columns:
        x_mkt = base["Close_mkt"].to_numpy(dtype=float)
        ma_mkt_arr = _fast_ma(x_mkt, int(market_ma_period))

    ma_dict_sig = {}
    valid_periods = sorted(set([int(w) for w in ma_pool if w and w > 0]))
    for w in valid_periods:
        ma_dict_sig[w] = _fast_ma(x_sig, w)
        
    return base, x_sig, x_trd, ma_dict_sig, x_mkt, ma_mkt_arr

# --- 시그널 체크 (상세) ---
def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell,
                       use_market_filter=False, market_ticker="", market_ma_period=200, 
                       use_bollinger=False, bb_period=20, bb_std=2.0, bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    if df is None or df.empty: st.error("데이터 없음"); return
    
    has_market = "Close_mkt" in df.columns
    ma_buy = int(ma_buy)
    ma_sell = int(ma_sell)
    
    # 데이터 정리
    df = df.copy().sort_values("Date").reset_index(drop=True)
    last_row = df.iloc[-1]
    last_date = pd.to_datetime(last_row['Date'])
    
    # [검증 1] 날짜 체크 (장 시작 전이면 어제 데이터일 수 있음)
    diff_days = (datetime.datetime.now() - last_date).days
    if diff_days > 4:
        st.warning(f"⚠️ 데이터가 {diff_days}일 전({last_date.date()}) 기준입니다. 최신 데이터가 아닐 수 있습니다.")
    else:
        st.caption(f"📅 기준일: **{last_date.strftime('%Y-%m-%d')}** (마감 데이터)")

    # 지표 계산
    df["Close"] = pd.to_numeric(df["Close_sig"], errors="coerce") 
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()
    
    if has_market and use_market_filter:
        df["MA_MKT"] = df["Close_mkt"].rolling(int(market_ma_period)).mean()
    
    if use_bollinger:
        m, u, l = calculate_bollinger_bands(df["Close"], bb_period, bb_std)
        df["BB_UP"], df["BB_MID"], df["BB_LO"] = u, m, l

    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"] = df["Close"].rolling(int(ma_compare_short)).mean()
        df["MA_LONG"] = df["Close"].rolling(int(ma_compare_long)).mean()
    
    i = len(df) - 1
    
    # [검증 2] 인덱스 안전장치
    max_offset = max(int(offset_cl_buy), int(offset_ma_buy), int(offset_cl_sell), int(offset_ma_sell), int(offset_compare_short), int(offset_compare_long))
    if i - max_offset < 0:
        st.error(f"데이터 부족 (최소 {max_offset}일 필요)"); return
        
    try:
        # 시장 필터 체크
        market_ok = True
        if has_market and use_market_filter:
            market_ok = df["Close_mkt"].iloc[i] > df["MA_MKT"].iloc[i]

        cl_b = float(df["Close"].iloc[i - int(offset_cl_buy)])
        cl_s = float(df["Close"].iloc[i - int(offset_cl_sell)])
        
        buy_ok, sell_ok = False, False
        cond_str, sell_cond_str = "", ""

        # 볼린저 밴드 로직
        if use_bollinger:
            bb_u, bb_m, bb_l = float(df["BB_UP"].iloc[i]), float(df["BB_MID"].iloc[i]), float(df["BB_LO"].iloc[i])
            prev_cl = float(df["Close"].iloc[i-1]) # 돌파 확인용 전일 종가
            
            # 매수
            if "상단선" in str(bb_entry_type): 
                # 상단 돌파 (어제는 아래, 오늘은 위)
                buy_ok = prev_cl <= bb_u and cl_b > bb_u; cond_str = f"종가 > 상단 {bb_u:.2f} (돌파)"
            elif "하단선" in str(bb_entry_type): 
                # 하단 이탈
                buy_ok = cl_b < bb_l; cond_str = f"종가 < 하단 {bb_l:.2f}"
            else: 
                # 중심선 돌파
                buy_ok = prev_cl <= bb_m and cl_b > bb_m; cond_str = f"종가 > 중심 {bb_m:.2f} (돌파)"

            # 매도
            if sell_operator == "OFF":
                sell_ok = False
                sell_cond_str = "OFF"
            else:
                if "상단선" in str(bb_exit_type): sell_ok = cl_s < bb_u; sell_cond_str = f"종가 < 상단 {bb_u:.2f}"
                elif "하단선" in str(bb_exit_type): sell_ok = cl_s < bb_l; sell_cond_str = f"종가 < 하단 {bb_l:.2f}"
                else: sell_ok = cl_s < bb_m; sell_cond_str = f"종가 < 중심 {bb_m:.2f}"
        
        # 이평선 로직
        else:
            ma_b = float(df["MA_BUY"].iloc[i - int(offset_ma_buy)])
            ma_s = float(df["MA_SELL"].iloc[i - int(offset_ma_sell)])
            
            # 추세 필터
            trend_ok = True
            if (use_trend_in_buy or use_trend_in_sell) and "MA_SHORT" in df.columns:
                trend_ok = df["MA_SHORT"].iloc[i - int(offset_compare_short)] >= df["MA_LONG"].iloc[i - int(offset_compare_long)]

            # 매수 판단
            buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
            
            # 매도 판단
            if sell_operator == "OFF":
                sell_ok = False
                sell_cond_str = "OFF"
            else:
                sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
                # 역추세 필터 (use_trend_in_sell): 정배열(trend_ok)이면 매도 안 함 -> 역배열(!trend_ok)일 때만 매도
                sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
                sell_cond_str = f"종가 {cl_s:.2f} {sell_operator} 이평 {ma_s:.2f}"
            
            buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
            cond_str = f"종가 {cl_b:.2f} {buy_operator} 이평 {ma_b:.2f}"

        final_buy = buy_ok and market_ok
        
        st.subheader(f"📌 시그널 진단")
        
        col1, col2 = st.columns(2)
        with col1:
             st.markdown(f"**🟢 매수 조건**")
             st.write(f"- 조건: {cond_str}")
             if use_trend_in_buy and not use_bollinger: st.write(f"- 추세: {'✅ 정배열' if trend_ok else '❌ 역배열'}")
             if use_market_filter: st.write(f"- 시장: {'✅ 상승장' if market_ok else '❌ 하락장'}")
             st.info(f"결과: {'✅ 진입' if final_buy else '⏸ 대기'}")
             
        with col2:
             st.markdown(f"**🔴 매도 조건**")
             st.write(f"- 조건: {sell_cond_str}")
             if use_trend_in_sell and not use_bollinger: st.write(f"- 역추세: {'✅ 역배열' if not trend_ok else '❌ 정배열'}")
             st.error(f"결과: {'✅ 청산' if sell_ok else '⏸ 보유'}")
        
        if final_buy and sell_ok:
            st.warning("⚠️ 매수/매도 신호 동시 발생! (전략 점검 권장)")

    except Exception as e: st.error(f"분석 중 오류: {e}")

def summarize_signal_today(df, p):
    if df is None or df.empty: return {"label": "N/A", "last_buy": "-"}
    try:
        # 데이터 정리
        df = df.copy().sort_values("Date").reset_index(drop=True)
        if len(df) < 60: return {"label": "데이터부족", "last_buy": "-"}
        
        last_idx = df.index[-1] # 가장 최근 (어제 or 오늘)
        
        # 파라미터 로드
        ma_buy = int(p.get("ma_buy", 20))
        off_ma_b = int(p.get("offset_ma_buy", 0))
        off_cl_b = int(p.get("offset_cl_buy", 0))
        buy_op = str(p.get("buy_operator", ">"))
        use_trend = bool(p.get("use_trend_in_buy", False))
        
        ma_s = int(p.get("ma_compare_short", 0) or 0)
        ma_l = int(p.get("ma_compare_long", 0) or 0)
        off_s = int(p.get("offset_compare_short", 0))
        off_l = int(p.get("offset_compare_long", 0))
        
        closes = pd.to_numeric(df["Close"], errors='coerce')
        
        # 매수 조건 체크
        # 1. 이평선 값
        ma_val = closes.rolling(ma_buy).mean().iloc[last_idx - off_ma_b]
        cl_val = closes.iloc[last_idx - off_cl_b]
        
        is_buy = False
        if buy_op == ">": is_buy = ma_val > cl_val
        elif buy_op == "<": is_buy = ma_val < cl_val
        
        # 2. 추세 필터
        if use_trend and ma_s > 0 and ma_l > 0:
            tr_s = closes.rolling(ma_s).mean().iloc[last_idx - off_s]
            tr_l = closes.rolling(ma_l).mean().iloc[last_idx - off_l]
            if tr_s <= tr_l: is_buy = False
            
        label = "🔵 BUY" if is_buy else "⚪ WAIT"
        last_buy_date = df['Date'].iloc[last_idx].strftime("%m-%d") if is_buy else "-"
        
        return {"label": label, "last_buy": last_buy_date}
        
    except: return {"label": "Error", "last_buy": "-"}

# --- 백테스트 함수 (기존 로직 유지) ---
def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70,
                  use_market_filter=False, x_mkt=None, ma_mkt_arr=None,
                  use_bollinger=False, bb_period=20, bb_std=2.0, 
                  bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈",
                  use_atr_stop=False, atr_multiplier=2.0):
    
    n = len(base)
    if n == 0: return {}
    
    ma_buy_arr = ma_dict_sig.get(int(ma_buy))
    ma_sell_arr = ma_dict_sig.get(int(ma_sell))
    
    # RSI
    rsi_arr = calculate_indicators(x_sig, int(rsi_period)) if use_rsi_filter else None
    
    # ATR
    atr_arr = base["ATR"].to_numpy(dtype=float) if use_atr_stop and "ATR" in base.columns else np.zeros(n)
    
    # 볼린저 밴드
    bb_up, bb_mid, bb_lo = None, None, None
    if use_bollinger: bb_mid, bb_up, bb_lo = calculate_bollinger_bands(x_sig, bb_period, bb_std)

    idx0 = 60 # 넉넉하게
    xC_trd = x_trd
    cash, position, hold_days, entry_price = float(initial_cash), 0.0, 0, 0.0
    logs, asset_curve = [], []

    def _fill(px, type): return px * (1 + (slip_bps + fee_bps)/10000.0) if type=='buy' else px * (1 - (slip_bps + fee_bps)/10000.0)

    for i in range(idx0, n):
        just_bought = False
        exec_price, signal, reason, reason_detail = None, "HOLD", None, ""
        close_today = xC_trd[i]
        # 매매 티커 데이터 (Open, High, Low)
        open_today = base["Open_trd"].iloc[i]
        low_today = base["Low_trd"].iloc[i]
        high_today = base["High_trd"].iloc[i]

        try:
            cl_b = x_sig[i - int(offset_cl_buy)]
            # ma_b는 ma_buy_arr에서 가져옴
        except: 
            asset_curve.append(cash + position * close_today)
            continue

        buy_cond, sell_cond = False, False
        buy_msg, sell_msg = "", "" 

        # 1. 시그널 로직
        if use_bollinger:
            idx_b, idx_s = i - int(offset_cl_buy), i - int(offset_cl_sell)
            
            if "상단선" in str(bb_entry_type): 
                buy_cond = cl_b > bb_up[idx_b]
                buy_msg = f"종가({cl_b:.2f}) > 상단({bb_up[idx_b]:.2f})"
            elif "하단선" in str(bb_entry_type): 
                buy_cond = cl_b < bb_lo[idx_b]
                buy_msg = f"종가({cl_b:.2f}) < 하단({bb_lo[idx_b]:.2f})"
            else: 
                buy_cond = cl_b > bb_mid[idx_b]
                buy_msg = f"종가({cl_b:.2f}) > 중심({bb_mid[idx_b]:.2f})"

            if "상단선" in str(bb_exit_type): 
                sell_cond = x_sig[i-int(offset_cl_sell)] < bb_up[idx_s]
                sell_msg = "종가 < 상단"
            elif "하단선" in str(bb_exit_type): 
                sell_cond = x_sig[i-int(offset_cl_sell)] < bb_lo[idx_s]
                sell_msg = "종가 < 하단"
            else: 
                sell_cond = x_sig[i-int(offset_cl_sell)] < bb_mid[idx_s]
                sell_msg = "종가 < 중심"
        else:
            # 이평선 로직
            ma_b = ma_buy_arr[i - int(offset_ma_buy)]
            ma_s = ma_sell_arr[i - int(offset_ma_sell)]
            
            # 추세 필터 확인
            t_ok = True
            if (use_trend_in_buy or use_trend_in_sell) and ma_compare_short and ma_compare_long:
                s_val = ma_dict_sig[int(ma_compare_short)][i-int(offset_compare_short)]
                l_val = ma_dict_sig[int(ma_compare_long)][i-int(offset_compare_long)]
                t_ok = s_val >= l_val

            if buy_operator == ">":
                buy_cond = (cl_b > ma_b)
                buy_msg = f"종가({cl_b:.2f}) > 이평({ma_b:.2f})"
            else:
                buy_cond = (cl_b < ma_b)
                buy_msg = f"종가({cl_b:.2f}) < 이평({ma_b:.2f})"
            
            if use_trend_in_buy and not t_ok: 
                buy_cond = False
                buy_msg += " (추세필터거부)"

            if sell_operator == "OFF":
                sell_cond = False
            else:
                cl_s = x_sig[i - int(offset_cl_sell)]
                if sell_operator == "<":
                    sell_cond = (cl_s < ma_s)
                    sell_msg = f"종가({cl_s:.2f}) < 이평({ma_s:.2f})"
                else:
                    sell_cond = (cl_s > ma_s)
                    sell_msg = f"종가({cl_s:.2f}) > 이평({ma_s:.2f})"
                
                if use_trend_in_sell and t_ok: 
                    sell_cond = False
                    sell_msg += " (역추세필터거부)"

        if buy_cond and use_rsi_filter:
            if rsi_arr[i-1] > rsi_max: 
                buy_cond = False
                buy_msg += f" (RSI 과열 {rsi_arr[i-1]:.1f})"
        
        if buy_cond and use_market_filter and x_mkt is not None:
            if x_mkt[i] < ma_mkt_arr[i]: 
                buy_cond = False
                buy_msg += " (시장하락장)"

        stop_hit, take_hit = False, False
        sold_today = False 

        # 3. 포지션 관리
        if position > 0:
            current_stop_price = 0.0
            atr_info = ""
            
            # ATR 손절
            if use_atr_stop and atr_arr[i-hold_days] > 0: 
                 entry_idx = i - hold_days
                 entry_atr = atr_arr[entry_idx]
                 current_stop_price = entry_price - (entry_atr * float(atr_multiplier))
                 atr_info = f"(ATR {atr_multiplier}배)"
            # % 손절
            elif stop_loss_pct > 0:
                current_stop_price = entry_price * (1 - stop_loss_pct / 100)
                atr_info = f"(-{stop_loss_pct}%)"
            
            # 손절 실행 (장중 저가 기준)
            if current_stop_price > 0 and low_today <= current_stop_price:
                stop_hit = True
                exec_price = open_today if open_today < current_stop_price else current_stop_price
                reason_detail = f"손절가 {current_stop_price:.2f} 도달 {atr_info}"
            
            # 익절 실행 (장중 고가 기준)
            if take_profit_pct > 0 and not stop_hit:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_today >= tp_price: 
                    take_hit = True
                    exec_price = open_today if open_today > tp_price else tp_price
                    reason_detail = f"익절가 {tp_price:.2f} 도달"

            if stop_hit or take_hit:
                if not stop_hit and not take_hit: exec_price = close_today 
                cash = position * _fill(exec_price, 'sell')
                r_type = "손절" if stop_hit else "익절"
                if stop_hit and use_atr_stop: r_type = "ATR손절"
                position, signal, reason, entry_price = 0.0, "SELL", r_type, 0.0
                sold_today = True

        if position > 0 and signal == "HOLD":
            if sell_cond and hold_days >= int(min_hold_days):
                exec_price = close_today
                cash = position * _fill(exec_price, 'sell')
                position, signal, reason, entry_price = 0.0, "SELL", "전략매도", 0.0
                reason_detail = sell_msg
                sold_today = True

        elif position == 0 and not sold_today:
            if buy_cond:
                exec_price = close_today
                position = cash / _fill(exec_price, 'buy')
                cash, signal, reason, just_bought, entry_price = 0.0, "BUY", "전략매수", True, exec_price
                reason_detail = buy_msg

        hold_days = hold_days + 1 if position > 0 and not just_bought else 0
        total = cash + (position * close_today)
        asset_curve.append(total)
        
        if signal != "HOLD":
            logs.append({
                "날짜": base["Date"].iloc[i], "종가": close_today, "신호": signal, 
                "체결가": exec_price, "자산": total, "이유": reason, 
                "상세내용": reason_detail, "손절발동": stop_hit, "익절발동": take_hit
            })

    if not logs: return {}
    s = pd.Series(asset_curve)
    
    g_profit, g_loss, wins = 0, 0, 0
    last_buy_price = None
    for r in logs:
        if r['신호'] == 'BUY': last_buy_price = r['체결가']
        elif r['신호'] == 'SELL' and last_buy_price:
            pnl = (r['체결가'] - last_buy_price) / last_buy_price
            if pnl > 0: wins += 1; g_profit += pnl
            else: g_loss += abs(pnl)
            last_buy_price = None
            
    total_sells = len([l for l in logs if l['신호']=='SELL'])
    pf = (g_profit / g_loss) if g_loss > 0 else 999.0
    win_rate = (wins / total_sells * 100) if total_sells > 0 else 0.0

    return {
        "수익률 (%)": round((asset_curve[-1] - initial_cash)/initial_cash*100, 2),
        "MDD (%)": round(((s - s.cummax()) / s.cummax()).min() * 100, 2),
        "승률 (%)": round(win_rate, 2),
        "Profit Factor": round(pf, 2),
        "총 매매 횟수": total_sells,
        "매매 로그": logs,
        "차트데이터": {"ma_buy_arr": ma_buy_arr[idx0:], "ma_sell_arr": ma_sell_arr[idx0:], "base": base.iloc[idx0:].reset_index(drop=True), "bb_up": bb_up[idx0:] if use_bollinger else None, "bb_lo": bb_lo[idx0:] if use_bollinger else None}
    }

def auto_search_train_test(signal_ticker, trade_ticker, start_date, end_date, split_ratio, choices_dict, n_trials=50, initial_cash=5000000, fee_bps=0, slip_bps=0, strategy_behavior="1", min_hold_days=0, constraints=None, **kwargs):
    ma_pool = set([5, 10, 20, 60, 120])
    for k in ["ma_buy", "ma_sell", "ma_compare_short", "ma_compare_long"]:
        for v in choices_dict.get(k, []):
            try:
                if int(v) > 0: ma_pool.add(int(v))
            except: pass
            
    base_full, x_sig_full, x_trd_full, ma_dict, _, _ = prepare_base(signal_ticker, trade_ticker, "", start_date, end_date, list(ma_pool))
    if base_full is None: return pd.DataFrame()
    
    split_idx = int(len(base_full) * split_ratio)
    base_tr, base_te = base_full.iloc[:split_idx].reset_index(drop=True), base_full.iloc[split_idx:].reset_index(drop=True)
    x_sig_tr, x_sig_te = x_sig_full[:split_idx], x_sig_full[split_idx:]
    x_trd_tr, x_trd_te = x_trd_full[:split_idx], x_trd_full[split_idx:]
    
    results = []
    defaults = {"ma_buy": 50, "ma_sell": 10, "offset_ma_buy": 0, "offset_ma_sell": 0, "offset_cl_buy":0, "offset_cl_sell":0, "buy_operator":">", "sell_operator":"<"}
    constraints = constraints or {}
    min_tr = constraints.get("min_trades", 0)
    min_wr = constraints.get("min_winrate", 0)
    limit_mdd = constraints.get("limit_mdd", 0)
    min_train_r = constraints.get("min_train_ret", -999.0)
    min_test_r = constraints.get("min_test_ret", -999.0)

    for _ in range(int(n_trials)):
        p = {}
        for k in choices_dict.keys():
            arr = choices_dict[k]
            p[k] = random.choice(arr) if arr else defaults.get(k)
        
        common_args = {
            "ma_dict_sig": ma_dict,
            "ma_buy": int(p.get('ma_buy', 50)), "offset_ma_buy": int(p.get('offset_ma_buy', 0)),
            "ma_sell": int(p.get('ma_sell', 10)), "offset_ma_sell": int(p.get('offset_ma_sell', 0)),
            "offset_cl_buy": int(p.get('offset_cl_buy', 0)), "offset_cl_sell": int(p.get('offset_cl_sell', 0)),
            "ma_compare_short": int(p.get('ma_compare_short')) if p.get('ma_compare_short') else 0,
            "ma_compare_long": int(p.get('ma_compare_long')) if p.get('ma_compare_long') else 0,
            "offset_compare_short": int(p.get('offset_compare_short', 0)), "offset_compare_long": int(p.get('offset_compare_long', 0)),
            "initial_cash": initial_cash, "stop_loss_pct": float(p.get('stop_loss_pct', 0)), "take_profit_pct": float(p.get('take_profit_pct', 0)),
            "strategy_behavior": strategy_behavior, "min_hold_days": min_hold_days, "fee_bps": fee_bps, "slip_bps": slip_bps,
            "use_trend_in_buy": p.get('use_trend_in_buy', True), "use_trend_in_sell": p.get('use_trend_in_sell', False),
            "buy_operator": p.get('buy_operator', '>'), "sell_operator": p.get('sell_operator', '<'),
            "use_atr_stop": p.get('use_atr_stop', False), "atr_multiplier": p.get('atr_multiplier', 2.0)
        }

        res_full = backtest_fast(base_full, x_sig_full, x_trd_full, **common_args)
        if not res_full: continue
        
        if res_full.get('총 매매 횟수', 0) < min_tr: continue
        if res_full.get('승률 (%)', 0) < min_wr: continue
        if limit_mdd > 0 and res_full.get('MDD (%)', 0) < -abs(limit_mdd): continue

        res_tr = backtest_fast(base_tr, x_sig_tr, x_trd_tr, **common_args)
        if res_tr.get('수익률 (%)', -999) < min_train_r: continue

        res_te = backtest_fast(base_te, x_sig_te, x_trd_te, **common_args)
        if res_te.get('수익률 (%)', -999) < min_test_r: continue

        row = {
            "Full_수익률(%)": res_full.get('수익률 (%)'), "Full_MDD(%)": res_full.get('MDD (%)'), "Full_승률(%)": res_full.get('승률 (%)'), "Full_총매매": res_full.get('총 매매 횟수'),
            "Test_수익률(%)": res_te.get('수익률 (%)'), "Test_MDD(%)": res_te.get('MDD (%)'),
            "Train_수익률(%)": res_tr.get('수익률 (%)'),
            "ma_buy": p.get('ma_buy'), "offset_ma_buy": p.get('offset_ma_buy'), "offset_cl_buy": p.get('offset_cl_buy'), "buy_operator": p.get('buy_operator'),
            "ma_sell": p.get('ma_sell'), "offset_ma_sell": p.get('offset_ma_sell'), "offset_cl_sell": p.get('offset_cl_sell'), "sell_operator": p.get('sell_operator'),
            "use_trend_in_buy": p.get('use_trend_in_buy'), "use_trend_in_sell": p.get('use_trend_in_sell'),
            "ma_compare_short": p.get('ma_compare_short'), "ma_compare_long": p.get('ma_compare_long'), "offset_compare_short": p.get('offset_compare_short'), "offset_compare_long": p.get('offset_compare_long'),
            "stop_loss_pct": p.get('stop_loss_pct'), "take_profit_pct": p.get('take_profit_pct'),
            "use_atr_stop": p.get('use_atr_stop'), "atr_multiplier": p.get('atr_multiplier')
        }
        results.append(row)
        
    return pd.DataFrame(results)

def apply_opt_params(row):
    try:
        updates = {
            "ma_buy": int(row["ma_buy"]), "offset_ma_buy": int(row["offset_ma_buy"]),
            "offset_cl_buy": int(row["offset_cl_buy"]), "buy_operator": str(row["buy_operator"]),
            "ma_sell": int(row["ma_sell"]), "offset_ma_sell": int(row["offset_ma_sell"]),
            "offset_cl_sell": int(row["offset_cl_sell"]), "sell_operator": str(row["sell_operator"]),
            "use_trend_in_buy": bool(row["use_trend_in_buy"]), "use_trend_in_sell": bool(row["use_trend_in_sell"]),
            "ma_compare_short": int(row["ma_compare_short"]) if not pd.isna(row["ma_compare_short"]) else 20,
            "ma_compare_long": int(row["ma_compare_long"]) if not pd.isna(row["ma_compare_long"]) else 50,
            "offset_compare_short": int(row["offset_compare_short"]),
            "offset_compare_long": int(row["offset_compare_long"]),
            "stop_loss_pct": float(row["stop_loss_pct"]),
            "take_profit_pct": float(row["take_profit_pct"]),
            "use_atr_stop": bool(row["use_atr_stop"]) if "use_atr_stop" in row else False,
            "atr_multiplier": float(row["atr_multiplier"]) if "atr_multiplier" in row else 2.0,
            "auto_run_trigger": True,
            "preset_name_selector": "직접 설정"
        }
        for k, v in updates.items(): st.session_state[k] = v
        st.toast("✅ 설정이 적용되었습니다! 백테스트 탭을 확인하세요.")
    except Exception as e: st.error(f"설정 적용 오류: {e}")

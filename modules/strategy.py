import pandas as pd
import numpy as np
import streamlit as st
import random
# [중요] 같은 폴더의 data_loader에서 get_data를 가져옴
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

# --- 데이터 준비 ---
@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, market_ticker, start_date, end_date, ma_pool, market_ma_period=200):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")
    
    if sig.empty or trd.empty: return None, None, None, None, None, None
    
    sig = sig.rename(columns={"Close": "Close_sig", "Open":"Open_sig", "High":"High_sig", "Low":"Low_sig"})[["Date", "Close_sig", "Open_sig", "High_sig", "Low_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    
    base = pd.merge(sig, trd, on="Date", how="inner")
    
    x_mkt, ma_mkt_arr = None, None
    if market_ticker:
        mkt = get_data(market_ticker, start_date, end_date).sort_values("Date")
        if not mkt.empty:
            mkt = mkt.rename(columns={"Close": "Close_mkt"})[["Date", "Close_mkt"]]
            base = pd.merge(base, mkt, on="Date", how="inner")
            
    base = base.dropna().reset_index(drop=True)
    
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)

    if "Close_mkt" in base.columns:
        x_mkt = base["Close_mkt"].to_numpy(dtype=float)
        ma_mkt_arr = _fast_ma(x_mkt, int(market_ma_period))

    ma_dict_sig = {}
    for w in sorted(set([int(w) for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)
        
    return base, x_sig, x_trd, ma_dict_sig, x_mkt, ma_mkt_arr

# --- 시그널 체크 ---
def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell,
                       use_market_filter=False, market_ticker="", market_ma_period=200, 
                       use_bollinger=False, bb_period=20, bb_std=2.0, bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    if df.empty: st.warning("데이터 없음"); return
    
    has_market = "Close_mkt" in df.columns
    ma_buy, ma_sell = int(ma_buy), int(ma_sell)
    
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close_sig"], errors="coerce") 
    df["MA_BUY"], df["MA_SELL"] = df["Close"].rolling(ma_buy).mean(), df["Close"].rolling(ma_sell).mean()
    
    if has_market and use_market_filter:
        df["MA_MKT"] = df["Close_mkt"].rolling(int(market_ma_period)).mean()
    
    if use_bollinger:
        m, u, l = calculate_bollinger_bands(df["Close"], bb_period, bb_std)
        df["BB_UP"], df["BB_MID"], df["BB_LO"] = u, m, l

    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"], df["MA_LONG"] = df["Close"].rolling(int(ma_compare_short)).mean(), df["Close"].rolling(int(ma_compare_long)).mean()
    
    i = len(df) - 1
    try:
        if i - max(int(offset_cl_buy), int(offset_ma_buy), int(offset_cl_sell), int(offset_ma_sell)) < 0:
            st.error("데이터 부족"); return
        
        market_ok = True
        if has_market and use_market_filter:
            market_ok = df["Close_mkt"].iloc[i] > df["MA_MKT"].iloc[i]

        cl_b = float(df["Close"].iloc[i - int(offset_cl_buy)])
        cl_s = float(df["Close"].iloc[i - int(offset_cl_sell)])
        ref_date = df["Date"].iloc[-1].strftime('%Y-%m-%d')
        
        buy_ok, sell_ok = False, False
        cond_str, sell_cond_str = "", ""

        if use_bollinger:
            bb_u, bb_m, bb_l = float(df["BB_UP"].iloc[i]), float(df["BB_MID"].iloc[i]), float(df["BB_LO"].iloc[i])
            if "상단선" in bb_entry_type: buy_ok = cl_b > bb_u; cond_str = f"종가 > 상단 {bb_u:.2f}"
            elif "하단선" in bb_entry_type: buy_ok = cl_b < bb_l; cond_str = f"종가 < 하단 {bb_l:.2f}"
            else: buy_ok = cl_b > bb_m; cond_str = f"종가 > 중심 {bb_m:.2f}"

            if "상단선" in bb_exit_type: sell_ok = cl_s < bb_u; sell_cond_str = f"종가 < 상단 {bb_u:.2f}"
            elif "하단선" in bb_exit_type: sell_ok = cl_s < bb_l; sell_cond_str = f"종가 < 하단 {bb_l:.2f}"
            else: sell_ok = cl_s < bb_m; sell_cond_str = f"종가 < 중심 {bb_m:.2f}"
        else:
            ma_b = float(df["MA_BUY"].iloc[i - int(offset_ma_buy)])
            ma_s = float(df["MA_SELL"].iloc[i - int(offset_ma_sell)])
            trend_ok = True
            if (use_trend_in_buy or use_trend_in_sell) and "MA_SHORT" in df.columns:
                trend_ok = df["MA_SHORT"].iloc[i - int(offset_compare_short)] >= df["MA_LONG"].iloc[i - int(offset_compare_long)]

            buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
            sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
            
            buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
            sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
            cond_str = f"종가 {cl_b:.2f} {buy_operator} 이평 {ma_b:.2f}"
            sell_cond_str = f"종가 {cl_s:.2f} {sell_operator} 이평 {ma_s:.2f}"

        final_buy = buy_ok and market_ok
        st.subheader(f"📌 시그널 ({ref_date})")
        st.write(f"💡 매수({bb_entry_type if use_bollinger else '이평'}): {cond_str} → {'✅' if buy_ok else '❌'}")
        if buy_ok and not market_ok: st.warning("⚠️ 시장 필터 미충족")
        st.write(f"💡 매도: {sell_cond_str} → {'✅' if sell_ok else '❌'}")
        
        if final_buy: st.success("🚀 매수 진입 (종가)")
        elif sell_ok: st.error("💧 매도 청산 (종가)")
        else: st.info("⏸ 관망")

    except Exception as e: st.error(f"오류: {e}")

def summarize_signal_today(df, p):
    if df is None or df.empty: return {"label": "N/A", "last_buy": "-", "last_sell": "-", "last_hold": "-"}
    return {"label": "확인필요", "last_buy": "-", "last_sell": "-", "last_hold": "-"}

# --- 백테스트 함수 ---
def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70,
                  use_market_filter=False, x_mkt=None, ma_mkt_arr=None,
                  use_bollinger=False, bb_period=20, bb_std=2.0, 
                  bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    n = len(base)
    if n == 0: return {}
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(int(ma_buy)), ma_dict_sig.get(int(ma_sell))
    ma_s_arr = ma_dict_sig.get(int(ma_compare_short)) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(int(ma_compare_long)) if ma_compare_long else None
    rsi_arr = calculate_indicators(x_sig, int(rsi_period)) if use_rsi_filter else None
    
    bb_up, bb_mid, bb_lo = None, None, None
    if use_bollinger: bb_mid, bb_up, bb_lo = calculate_bollinger_bands(x_sig, bb_period, bb_std)

    idx0 = 50
    xC_trd = x_trd
    cash, position, hold_days, entry_price = float(initial_cash), 0.0, 0, 0.0
    logs, asset_curve = [], []

    def _fill(px, type): return px * (1 + (slip_bps + fee_bps)/10000.0) if type=='buy' else px * (1 - (slip_bps + fee_bps)/10000.0)

    for i in range(idx0, n):
        just_bought = False
        exec_price, signal, reason = None, "HOLD", None
        close_today = xC_trd[i]
        open_today, low_today, high_today = base["Open_trd"].iloc[i], base["Low_trd"].iloc[i], base["High_trd"].iloc[i]

        try:
            cl_b, ma_b = x_sig[i - int(offset_cl_buy)], ma_buy_arr[i - int(offset_ma_buy)]
            cl_s, ma_s = x_sig[i - int(offset_cl_sell)], ma_sell_arr[i - int(offset_ma_sell)]
        except: asset_curve.append(cash + position * close_today); continue

        buy_cond, sell_cond = False, False

        if use_bollinger:
            idx_b = i - int(offset_cl_buy)
            idx_s = i - int(offset_cl_sell)
            
            if "상단선" in str(bb_entry_type): buy_cond = cl_b > bb_up[idx_b]
            elif "하단선" in str(bb_entry_type): buy_cond = cl_b < bb_lo[idx_b]
            else: buy_cond = cl_b > bb_mid[idx_b]

            if "상단선" in str(bb_exit_type): sell_cond = cl_s < bb_up[idx_s]
            elif "하단선" in str(bb_exit_type): sell_cond = cl_s < bb_lo[idx_s]
            else: sell_cond = cl_s < bb_mid[idx_s]
        else:
            t_ok = True
            if ma_s_arr is not None: t_ok = ma_s_arr[i-int(offset_compare_short)] >= ma_l_arr[i-int(offset_compare_long)]
            buy_cond = ((cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)) and (t_ok if use_trend_in_buy else True)
            sell_cond = ((cl_s < ma_s) if sell_operator == "<" else (cl_s > ma_s)) and ((not t_ok) if use_trend_in_sell else True)

        if buy_cond and use_rsi_filter and rsi_arr[i-1] > rsi_max: buy_cond = False
        if buy_cond and use_market_filter and x_mkt[i] < ma_mkt_arr[i]: buy_cond = False

        stop_hit, take_hit = False, False
        sold_today = False 

        if position > 0:
            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if low_today <= sl_price: 
                    stop_hit = True
                    exec_price = open_today if open_today < sl_price else sl_price
            
            if take_profit_pct > 0 and not stop_hit:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_today >= tp_price: 
                    take_hit = True
                    exec_price = open_today if open_today > tp_price else tp_price

            if stop_hit or take_hit:
                if not stop_hit and not take_hit: exec_price = close_today 
                cash = position * _fill(exec_price, 'sell')
                position, signal, reason, entry_price = 0.0, "SELL", "손절" if stop_hit else "익절", 0.0
                sold_today = True

        if position > 0 and signal == "HOLD":
            if sell_cond and hold_days >= int(min_hold_days):
                exec_price = close_today
                cash = position * _fill(exec_price, 'sell')
                position, signal, reason, entry_price = 0.0, "SELL", "전략매도", 0.0
                sold_today = True

        elif position == 0 and not sold_today:
            if buy_cond:
                exec_price = close_today
                position = cash / _fill(exec_price, 'buy')
                cash, signal, reason, just_bought, entry_price = 0.0, "BUY", "전략매수", True, exec_price

        hold_days = hold_days + 1 if position > 0 and not just_bought else 0
        total = cash + (position * close_today)
        asset_curve.append(total)
        
        logs.append({
            "날짜": base["Date"].iloc[i], "종가": close_today, "신호": signal, 
            "체결가": exec_price, "자산": total, "이유": reason,
            "손절발동": stop_hit, "익절발동": take_hit
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
            "buy_operator": p.get('buy_operator', '>'), "sell_operator": p.get('sell_operator', '<')
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
            "stop_loss_pct": p.get('stop_loss_pct'), "take_profit_pct": p.get('take_profit_pct')
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
            "auto_run_trigger": True,
            "preset_name_selector": "직접 설정"
        }
        for k, v in updates.items(): st.session_state[k] = v
        st.toast("✅ 설정이 적용되었습니다! 백테스트 탭을 확인하세요.")
    except Exception as e: st.error(f"설정 적용 오류: {e}")
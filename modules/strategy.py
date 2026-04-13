import pandas as pd
import numpy as np
import streamlit as st
import random
from .data_loader import get_data
import optuna

# Optuna가 실행할 목적 함수
def optuna_objective(trial, base_full, x_sig_full, x_trd_full, ma_dict, initial_cash, fee_bps, slip_bps, strategy_behavior, min_hold_days):
    
    # 🎯 [핵심] 사용자가 원하는 "딱 떨어지는 숫자 리스트" 만들기
    # 1을 포함하고, 5부터 120까지 5 단위로 리스트 생성 -> [1, 5, 10, 15, ..., 120]
    ma_list = [1] + list(range(5, 121, 5))
    offset_list = [1] + list(range(5, 61, 5))

    p = {
        # 1. 매수 조건 (만들어둔 ma_list 안에서만 고르도록 AI에게 지시)
        "ma_buy": trial.suggest_categorical("ma_buy", ma_list),
        "offset_ma_buy": trial.suggest_categorical("offset_ma_buy", offset_list),
        "offset_cl_buy": trial.suggest_categorical("offset_cl_buy", offset_list),
        "buy_operator": trial.suggest_categorical("buy_operator", [">", "<"]),
        
        # 2. 매도 조건 (역시 ma_list 안에서만 고름)
        "ma_sell": trial.suggest_categorical("ma_sell", ma_list),
        "offset_ma_sell": trial.suggest_categorical("offset_ma_sell", offset_list),
        "offset_cl_sell": trial.suggest_categorical("offset_cl_sell", offset_list),
        "sell_operator": trial.suggest_categorical("sell_operator", ["<", ">", "OFF"]),
        
        # 3. 추세 필터 
        # (여기는 시작점이 5, 60으로 5와 10의 배수라 step만 줘도 5, 10, 15...로 깔끔하게 떨어집니다!)
        "use_trend_in_buy": trial.suggest_categorical("use_trend_in_buy", [True, False]),
        "use_trend_in_sell": trial.suggest_categorical("use_trend_in_sell", [True, False]),
        "ma_compare_short": trial.suggest_categorical("ma_compare_short", ma_list),
        "ma_compare_long": trial.suggest_categorical("ma_compare_long", ma_list),
        "offset_compare_short": trial.suggest_categorical("offset_compare_short", offset_list),
        "offset_compare_long": trial.suggest_categorical("offset_compare_long", offset_list),
        
        # 4. 리스크 관리 (손/익절)
        "stop_loss_pct": trial.suggest_float("stop_loss_pct", 15, 35, step=5),
        "take_profit_pct": trial.suggest_float("take_profit_pct", 0, 30.0, step=5),
        
        # 5. ATR 동적 손절
        "use_atr_stop": trial.suggest_categorical("use_atr_stop", [True, False]),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 2.0, 5.0, step=1)
    }

    # 🛑 [AI 속도 향상] 단기 이평선이 장기 이평선보다 크거나 같으면 논리 오류이므로 즉시 폐기(Pruned)
    if p["use_trend_in_buy"] or p["use_trend_in_sell"]:
        if p["ma_compare_short"] >= p["ma_compare_long"]:
            raise optuna.TrialPruned()

    # AI가 제안한 파라미터로 백테스트 실행
    res = backtest_fast(
        base_full, x_sig_full, x_trd_full, ma_dict,
        initial_cash=initial_cash, fee_bps=fee_bps, slip_bps=slip_bps,
        strategy_behavior=strategy_behavior, min_hold_days=min_hold_days,
        **p 
    )

    # 평가: 1년에 5번도 매매 안 하는 우연의 일치는 걸러냄 (-999점 부여)
    if not res or res.get("총 매매 횟수", 0) < 5:
        return -999.0
        
    return res.get("수익률 (%)", -999.0)

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
    import datetime
    import pandas as pd
    
    # 1. 종료일 하루 잘림 방지 (+1일)
    end_date_adj = pd.to_datetime(end_date) + datetime.timedelta(days=1)
    end_date_str = end_date_adj.strftime("%Y-%m-%d")
    
    sig = get_data(signal_ticker, start_date, end_date_str)
    trd = get_data(trade_ticker,  start_date, end_date_str)
    
    if sig is None or sig.empty or trd is None or trd.empty: 
        return None, None, None, None, None, None
        
    # 🛡️ [핵심 방어막 1] 날짜에서 시간(15:30:00 등)을 완전히 제거하여 00:00:00으로 통일
    sig['Date'] = pd.to_datetime(sig['Date']).dt.normalize()
    trd['Date'] = pd.to_datetime(trd['Date']).dt.normalize()

    # 🛡️ [핵심 방어막 2] yfinance 최신 날짜 중복(가짜 실시간 캔들) 버그 완벽 제거!
    # 같은 날짜가 여러 개면 가장 마지막(최신) 값 딱 1개만 남기고 다 지웁니다.
    sig = sig.drop_duplicates(subset=['Date'], keep='last')
    trd = trd.drop_duplicates(subset=['Date'], keep='last')
    
    sig = sig.sort_values("Date")
    trd = trd.sort_values("Date")
    
    # 사용자가 지정한 종료일까지만 정확히 자르기
    target_end_date = pd.to_datetime(end_date)
    sig = sig[sig['Date'] <= target_end_date]
    trd = trd[trd['Date'] <= target_end_date]
    
    # 주말(토=5, 일=6) 가짜 캔들 삭제
    sig = sig[~sig['Date'].dt.dayofweek.isin([5, 6])]
    trd = trd[~trd['Date'].dt.dayofweek.isin([5, 6])]

    # ATR 계산
    trd["ATR"] = calculate_atr(trd, period=14)

    sig = sig.rename(columns={"Close": "Close_sig", "Open":"Open_sig", "High":"High_sig", "Low":"Low_sig"})[["Date", "Close_sig", "Open_sig", "High_sig", "Low_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd", "ATR": "ATR"})
    
    base = pd.merge(sig, trd, on="Date", how="inner")
    
    x_mkt, ma_mkt_arr = None, None
    if market_ticker:
        mkt = get_data(market_ticker, start_date, end_date_str)
        if not mkt.empty:
            mkt['Date'] = pd.to_datetime(mkt['Date']).dt.normalize()
            mkt = mkt.drop_duplicates(subset=['Date'], keep='last') # 시장 데이터도 중복 제거
            mkt = mkt.sort_values("Date")
            mkt = mkt[mkt['Date'] <= target_end_date]
            mkt = mkt[~mkt['Date'].dt.dayofweek.isin([5, 6])]
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

# --- 시그널 체크 (상세) ---
def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell,
                       use_market_filter=False, market_ticker="", market_ma_period=200, 
                       use_bollinger=False, bb_period=20, bb_std=2.0, bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    if df is None or df.empty: st.error("데이터 없음"); return
    
    # 🛡️ [핵심] 문자열(False) 버그 완벽 차단 및 형변환
    def _bool(v): return str(v).strip().lower() in ['true', '1', 't', 'y']
    def _int(v, d=0):
        try: return int(float(v))
        except: return d
        
    use_trend_in_buy = _bool(use_trend_in_buy)
    use_trend_in_sell = _bool(use_trend_in_sell)
    ma_comp_s = _int(ma_compare_short)
    ma_comp_l = _int(ma_compare_long)
    off_comp_s = _int(offset_compare_short)
    off_comp_l = _int(offset_compare_long)

    # 🧹 [가짜 캔들 삭제됨] 순수하게 존재하는 데이터만 정렬
    df = df.copy().sort_values("Date").reset_index(drop=True)
    if "Close_sig" in df.columns: 
        df["Close"] = pd.to_numeric(df["Close_sig"], errors="coerce")
    else: 
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        
    last_row = df.iloc[-1]
    last_date = pd.to_datetime(last_row['Date'])
    
    import datetime
    diff_days = (datetime.datetime.now().date() - last_date.date()).days
    if diff_days >= 1:
        st.info(f"💡 장 시작 전입니다. **{last_date.strftime('%Y-%m-%d')} (전일 종가)** 기준으로 분석합니다.")
    else:
        st.caption(f"📅 기준일: **{last_date.strftime('%Y-%m-%d')}** (최신)")
    
    has_market = "Close_mkt" in df.columns
    ma_buy = _int(ma_buy, 20)
    ma_sell = _int(ma_sell, 10)
    
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()
    
    if has_market and use_market_filter:
        df["MA_MKT"] = df["Close_mkt"].rolling(_int(market_ma_period, 200)).mean()
    
    if use_bollinger:
        m, u, l = calculate_bollinger_bands(df["Close"], _int(bb_period, 20), float(bb_std))
        df["BB_UP"], df["BB_MID"], df["BB_LO"] = u, m, l

    if ma_comp_s > 0 and ma_comp_l > 0:
        df["MA_SHORT"] = df["Close"].rolling(ma_comp_s).mean()
        df["MA_LONG"] = df["Close"].rolling(ma_comp_l).mean()
    
    i = len(df) - 1
    try:
        if i - max(_int(offset_cl_buy), _int(offset_ma_buy), _int(offset_cl_sell), _int(offset_ma_sell), off_comp_s, off_comp_l) < 0:
            st.error("데이터 부족"); return
        
        market_ok = True
        if has_market and use_market_filter:
            market_ok = df["Close_mkt"].iloc[i] > df["MA_MKT"].iloc[i]

        cl_b = float(df["Close"].iloc[i - _int(offset_cl_buy)])
        cl_s = float(df["Close"].iloc[i - _int(offset_cl_sell)])
        ref_date = df["Date"].iloc[-1].strftime('%Y-%m-%d')
        
        buy_ok, sell_ok = False, False
        cond_str, sell_cond_str = "", ""

        if use_bollinger:
            bb_u, bb_m, bb_l = float(df["BB_UP"].iloc[i]), float(df["BB_MID"].iloc[i]), float(df["BB_LO"].iloc[i])
            if "상단선" in str(bb_entry_type): buy_ok = cl_b > bb_u; cond_str = f"종가 > 상단 {bb_u:.2f}"
            elif "하단선" in str(bb_entry_type): buy_ok = cl_b < bb_l; cond_str = f"종가 < 하단 {bb_l:.2f}"
            else: buy_ok = cl_b > bb_m; cond_str = f"종가 > 중심 {bb_m:.2f}"

            if sell_operator == "OFF":
                sell_ok = False
                sell_cond_str = "OFF (전략매도 끔)"
            else:
                if "상단선" in str(bb_exit_type): sell_ok = cl_s < bb_u; sell_cond_str = f"종가 < 상단 {bb_u:.2f}"
                elif "하단선" in str(bb_exit_type): sell_ok = cl_s < bb_l; sell_cond_str = f"종가 < 하단 {bb_l:.2f}"
                else: sell_ok = cl_s < bb_m; sell_cond_str = f"종가 < 중심 {bb_m:.2f}"
        else:
            ma_b = float(df["MA_BUY"].iloc[i - _int(offset_ma_buy)])
            ma_s = float(df["MA_SELL"].iloc[i - _int(offset_ma_sell)])
            
            # 🛡️ [추가] 추세 필터 판단 결과 및 UI 화면 출력용 텍스트 생성
            trend_ok = True
            t_str_debug = ""
            if (use_trend_in_buy or use_trend_in_sell):
                if "MA_SHORT" in df.columns and "MA_LONG" in df.columns:
                    s_val = df["MA_SHORT"].iloc[i - off_comp_s]
                    l_val = df["MA_LONG"].iloc[i - off_comp_l]
                    if pd.isna(s_val) or pd.isna(l_val):
                        trend_ok = False
                        t_str_debug = " [추세 데이터부족]"
                    else:
                        trend_ok = (s_val >= l_val)
                        t_str_debug = f" [추세: 단기{s_val:.2f} {'≥' if trend_ok else '<'} 장기{l_val:.2f}]"
                else:
                    trend_ok = False

            buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
            
            if sell_operator == "OFF":
                sell_ok = False
                sell_cond_str = "OFF (전략매도 끔)"
            else:
                sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
                sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
                sell_cond_str = f"종가 {cl_s:.2f} {sell_operator} 이평 {ma_s:.2f}"
                if use_trend_in_sell: sell_cond_str += t_str_debug
            
            buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
            cond_str = f"종가 {cl_b:.2f} {buy_operator} 이평 {ma_b:.2f}"
            if use_trend_in_buy: cond_str += t_str_debug

        final_buy = buy_ok and market_ok
        st.subheader(f"📌 시그널 ({ref_date})")
        st.write(f"💡 매수({bb_entry_type if use_bollinger else '이평'}): {cond_str} → {'✅' if buy_ok else '❌'}")
        if buy_ok and not market_ok: st.warning("⚠️ 시장 필터 미충족")
        st.write(f"💡 매도: {sell_cond_str} → {'✅' if sell_ok else '❌'}")
        
        if final_buy and sell_ok:
            st.warning("⚠️ 매수/매도 신호 중복 (전략 점검 필요)")
        elif final_buy:
            st.success("🚀 매수 진입 (종가)")
        elif sell_ok:
            st.error("💧 매도 청산 (종가)")
        else:
            st.info("⏸ 관망")

    except Exception as e: st.error(f"오류: {e}")

# --- 프리셋 분석 (Tab 2) ---
def summarize_signal_today(df, p):
    if df is None or df.empty: return {"label": "N/A", "last_buy": "-", "last_sell": "-", "last_hold": "-"}
    try:
        # 🧹 [가짜 캔들 삭제됨] 순수하게 존재하는 데이터만 정렬
        df = df.copy().sort_values("Date").reset_index(drop=True)
        if "Close_sig" in df.columns: 
            df["Close"] = pd.to_numeric(df["Close_sig"], errors="coerce")
        else: 
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            
        if len(df) < 60: return {"label": "데이터부족", "last_buy": "-", "last_sell": "-", "last_hold": "-"}

        idx_now = len(df) - 1
        
        def _bool(v): return str(v).strip().lower() in ['true', '1', 't', 'y']
        def _int(v, d=0): 
            try: return int(float(v))
            except: return d

        ma_buy = _int(p.get("ma_buy", 20))
        ma_sell = _int(p.get("ma_sell", 10))
        off_ma_b = _int(p.get("offset_ma_buy", 0))
        off_cl_b = _int(p.get("offset_cl_buy", 0))
        off_ma_s = _int(p.get("offset_ma_sell", 0))
        off_cl_s = _int(p.get("offset_cl_sell", 0))
        
        buy_op = str(p.get("buy_operator", ">")).strip()
        sell_op = str(p.get("sell_operator", "<")).strip()
        
        use_trend_buy = _bool(p.get("use_trend_in_buy", False))
        use_trend_sell = _bool(p.get("use_trend_in_sell", False))
        ma_comp_s = _int(p.get("ma_compare_short", 0))
        ma_comp_l = _int(p.get("ma_compare_long", 0))
        off_comp_s = _int(p.get("offset_compare_short", 0))
        off_comp_l = _int(p.get("offset_compare_long", 0))
        use_bollinger = _bool(p.get("use_bollinger", False))
        
        if (use_trend_buy or use_trend_sell) and ma_comp_s > 0 and ma_comp_l > 0:
            df["MA_COMP_S"] = df["Close"].rolling(ma_comp_s).mean()
            df["MA_COMP_L"] = df["Close"].rolling(ma_comp_l).mean()

        if use_bollinger:
            bb_p = _int(p.get("bb_period", 20))
            try: bb_s = float(p.get("bb_std", 2.0))
            except: bb_s = 2.0
            _, u, l = calculate_bollinger_bands(df["Close"], bb_p, bb_s)
            mid = df["Close"].rolling(bb_p).mean()
            df["BB_UP"], df["BB_LO"], df["BB_MID"] = u, l, mid
        else:
            df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
            df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()

        last_buy_date, last_sell_date = "-", "-"
        debug_msg = "" 

        def _check(i, type_):
            nonlocal debug_msg
            if i < max(60, off_ma_b, off_cl_b, off_ma_s, off_cl_s): return False
            try:
                if type_ == 'sell' and sell_op == "OFF": return False

                trend_ok = True
                if (use_trend_buy or use_trend_sell) and "MA_COMP_S" in df.columns:
                    s_val = df["MA_COMP_S"].iloc[i - off_comp_s]
                    l_val = df["MA_COMP_L"].iloc[i - off_comp_l]
                    trend_ok = (s_val >= l_val)

                if use_bollinger:
                    bb_entry = str(p.get("bb_entry_type", ""))
                    bb_exit = str(p.get("bb_exit_type", ""))
                    cl = df["Close"].iloc[i - (off_cl_b if type_=='buy' else off_cl_s)]
                    if type_ == 'buy':
                        if "상단선" in bb_entry: return cl > df["BB_UP"].iloc[i-off_cl_b]
                        elif "하단선" in bb_entry: return cl < df["BB_LO"].iloc[i-off_cl_b]
                        else: return cl > df["BB_MID"].iloc[i-off_cl_b]
                    else:
                        if "상단선" in bb_exit: return cl < df["BB_UP"].iloc[i-off_cl_s]
                        elif "하단선" in bb_exit: return cl < df["BB_LO"].iloc[i-off_cl_s]
                        else: return cl < df["BB_MID"].iloc[i-off_cl_s]
                else:
                    cl = df["Close"].iloc[i - (off_cl_b if type_=='buy' else off_cl_s)]
                    ma = df["MA_BUY"].iloc[i - off_ma_b] if type_=='buy' else df["MA_SELL"].iloc[i - off_ma_s]
                    
                    if type_ == 'buy':
                        buy_cond = (cl > ma) if buy_op == ">" else (cl < ma)
                        res = (buy_cond and trend_ok) if use_trend_buy else buy_cond
                        
                        # [UI 피드백] 프리셋 화면에 왜 관망인지 이유 출력
                        if i == idx_now and not res:
                            t_info = f", 추세:{'✅' if trend_ok else '❌'}" if use_trend_buy else ""
                            debug_msg = f"(종가 vs 이평{t_info})"
                        return res
                    else:
                        sell_cond = (cl < ma) if sell_op == "<" else (cl > ma)
                        return (sell_cond and (not trend_ok)) if use_trend_sell else sell_cond
            except Exception as e: 
                return False

        is_buy_now = _check(idx_now, 'buy')
        is_sell_now = _check(idx_now, 'sell')
        
        label = f"관망 {debug_msg}".strip() if debug_msg else "관망"
        if is_buy_now and is_sell_now: label = "⚠️매수/매도 중복"
        elif is_buy_now: label = "매수진입"
        elif is_sell_now: label = "매도청산"
        
        search_range = min(365, len(df)-60)
        for k in range(search_range):
            curr_idx = idx_now - k
            d_str = df["Date"].iloc[curr_idx].strftime("%Y-%m-%d")
            if last_buy_date == "-" and _check(curr_idx, 'buy'): last_buy_date = d_str
            if last_sell_date == "-" and _check(curr_idx, 'sell'): last_sell_date = d_str
            if last_buy_date != "-" and last_sell_date != "-": break
        
        return {"label": label, "last_buy": last_buy_date, "last_sell": last_sell_date, "last_hold": "-"}
    except Exception as e: return {"label": f"오류:{e}", "last_buy": "-", "last_sell": "-", "last_hold": "-"}
        

# --- 백테스트 함수 (상세 로그 버전으로 교체됨) ---
def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70,
                  use_market_filter=False, x_mkt=None, ma_mkt_arr=None,
                  use_bollinger=False, bb_period=20, bb_std=2.0, 
                  bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈",
                  use_atr_stop=False, atr_multiplier=2.0):
    
    n = len(base)
    if n == 0: return {}
    
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(int(ma_buy)), ma_dict_sig.get(int(ma_sell))
    ma_s_arr = ma_dict_sig.get(int(ma_compare_short)) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(int(ma_compare_long)) if ma_compare_long else None
    rsi_arr = calculate_indicators(x_sig, int(rsi_period)) if use_rsi_filter else None
    atr_arr = base["ATR"].to_numpy(dtype=float) if "ATR" in base.columns else np.zeros(n)
    
    bb_up, bb_mid, bb_lo = None, None, None
    if use_bollinger: bb_mid, bb_up, bb_lo = calculate_bollinger_bands(x_sig, bb_period, bb_std)

    idx0 = 50
    xC_trd = x_trd
    cash, position, hold_days, entry_price = float(initial_cash), 0.0, 0, 0.0
    logs, asset_curve = [], []

    def _fill(px, type): return px * (1 + (slip_bps + fee_bps)/10000.0) if type=='buy' else px * (1 - (slip_bps + fee_bps)/10000.0)

    for i in range(idx0, n):
        just_bought = False
        exec_price, signal, reason, reason_detail = None, "HOLD", None, ""
        close_today = xC_trd[i]
        open_today, low_today, high_today = base["Open_trd"].iloc[i], base["Low_trd"].iloc[i], base["High_trd"].iloc[i]

        # 👇👇 [여기서부터 교체 시작] 👇👇
        
        # [핵심 수정] 현실적인 매매(T+1) 구현: 신호는 어제(i-1) 기준으로 판단 -> 매매는 오늘(i) 실행
        prev_i = i - 1
        
        try:
            # 모든 기준점(i)을 어제(prev_i)로 밀어버립니다.
            cl_b, ma_b = x_sig[prev_i - int(offset_cl_buy)], ma_buy_arr[prev_i - int(offset_ma_buy)]
            cl_s, ma_s = x_sig[prev_i - int(offset_cl_sell)], ma_sell_arr[prev_i - int(offset_ma_sell)]
        except: 
            asset_curve.append(cash + position * close_today)
            continue

        buy_cond, sell_cond = False, False
        buy_msg, sell_msg = "", "" 

        # 1. 기술적 지표 조건 판단 (전부 prev_i 기준)
        if use_bollinger:
            idx_b, idx_s = prev_i - int(offset_cl_buy), prev_i - int(offset_cl_sell)
            
            if "상단선" in str(bb_entry_type): 
                buy_cond = cl_b > bb_up[idx_b]
                buy_msg = f"어제종가({cl_b:.2f}) > 상단({bb_up[idx_b]:.2f})"
            elif "하단선" in str(bb_entry_type): 
                buy_cond = cl_b < bb_lo[idx_b]
                buy_msg = f"어제종가({cl_b:.2f}) < 하단({bb_lo[idx_b]:.2f})"
            else: 
                buy_cond = cl_b > bb_mid[idx_b]
                buy_msg = f"어제종가({cl_b:.2f}) > 중심({bb_mid[idx_b]:.2f})"

            if "상단선" in str(bb_exit_type): 
                sell_cond = cl_s < bb_up[idx_s]
                sell_msg = f"어제종가({cl_s:.2f}) < 상단({bb_up[idx_s]:.2f})"
            elif "하단선" in str(bb_exit_type): 
                sell_cond = cl_s < bb_lo[idx_s]
                sell_msg = f"어제종가({cl_s:.2f}) < 하단({bb_lo[idx_s]:.2f})"
            else: 
                sell_cond = cl_s < bb_mid[idx_s]
                sell_msg = f"어제종가({cl_s:.2f}) < 중심({bb_mid[idx_s]:.2f})"
        else:
            t_ok = True
            t_msg = ""
            if ma_s_arr is not None: 
                s_val = ma_s_arr[prev_i-int(offset_compare_short)]
                l_val = ma_l_arr[prev_i-int(offset_compare_long)]
                t_ok = s_val >= l_val
                t_msg = f" [추세:{'상승' if t_ok else '하락'}]"

            if buy_operator == ">":
                buy_cond = (cl_b > ma_b)
                buy_msg = f"어제종가({cl_b:.2f}) > 이평({ma_b:.2f})"
            else:
                buy_cond = (cl_b < ma_b)
                buy_msg = f"어제종가({cl_b:.2f}) < 이평({ma_b:.2f})"
            
            if use_trend_in_buy and not t_ok: 
                buy_cond = False
                buy_msg += " (추세필터거부)"

            if sell_operator == "OFF":
                sell_cond = False
                sell_msg = "매도조건 OFF"
            else:
                if sell_operator == "<":
                    sell_cond = (cl_s < ma_s)
                    sell_msg = f"어제종가({cl_s:.2f}) < 이평({ma_s:.2f})"
                else:
                    sell_cond = (cl_s > ma_s)
                    sell_msg = f"어제종가({cl_s:.2f}) > 이평({ma_s:.2f})"
                
                if use_trend_in_sell and t_ok: 
                    sell_cond = False
                    sell_msg += " (역추세필터거부)"

        if buy_cond and use_rsi_filter:
            if rsi_arr[prev_i] > rsi_max: 
                buy_cond = False
                buy_msg += f" (RSI 과열 {rsi_arr[prev_i]:.1f})"
        
        if buy_cond and use_market_filter:
            if x_mkt[prev_i] < ma_mkt_arr[prev_i]: 
                buy_cond = False
                buy_msg += f" (시장하락장 {x_mkt[prev_i]:.1f})"
                
        # 👆👆 [여기까지 교체 끝] 👆👆
        
        # 2. 매도 OFF 강제 적용
        if sell_operator == "OFF":
            sell_cond = False
            sell_msg = "OFF"

        stop_hit, take_hit = False, False
        sold_today = False 

        # 3. 포지션 관리 (진입/청산)
        if position > 0:
            current_stop_price = 0.0
            atr_info_str = ""
            
            if use_atr_stop and atr_arr[i-hold_days] > 0: 
                 entry_idx = i - hold_days
                 if entry_idx >= 0:
                     entry_atr = atr_arr[entry_idx]
                     current_stop_price = entry_price - (entry_atr * float(atr_multiplier))
                     atr_info_str = f"(ATR:{entry_atr:.2f}x{atr_multiplier})"
            elif stop_loss_pct > 0:
                current_stop_price = entry_price * (1 - stop_loss_pct / 100)
                atr_info_str = f"(-{stop_loss_pct}%)"
            
            if current_stop_price > 0 and low_today <= current_stop_price:
                stop_hit = True
                exec_price = open_today if open_today < current_stop_price else current_stop_price
                reason_detail = f"장중저가({low_today:.2f}) <= 손절가({current_stop_price:.2f}) {atr_info_str}"
            
            if take_profit_pct > 0 and not stop_hit:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_today >= tp_price: 
                    take_hit = True
                    exec_price = open_today if open_today > tp_price else tp_price
                    reason_detail = f"장중고가({high_today:.2f}) >= 익절가({tp_price:.2f})"

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
               
       # 🛡️ [수정] 마지막 5일은 매매를 안 했어도(HOLD) 강제로 로그 표에 '관망(디버그)'로 박제합니다!
        if signal != "HOLD": # or i >= n - 5:
            logs.append({
                "날짜": base["Date"].iloc[i], 
                "종가": close_today, 
                "신호": signal if signal != "HOLD" else "관망(디버그)", 
                "체결가": exec_price if exec_price is not None else close_today, 
                "자산": total, 
                "이유": reason if reason else "조건확인", 
                "상세내용": reason_detail if signal != "HOLD" else f"매수통과?:{buy_cond} | {buy_msg}", 
                "손절발동": stop_hit, 
                "익절발동": take_hit
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

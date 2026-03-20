# --- 시그널 체크 (상세) ---
def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell,
                       use_market_filter=False, market_ticker="", market_ma_period=200, 
                       use_bollinger=False, bb_period=20, bb_std=2.0, bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    if df is None or df.empty: st.error("데이터 없음"); return
    
    # 순수하게 데이터 정렬만 수행 (가짜 캔들 없음)
    df = df.copy().sort_values("Date").reset_index(drop=True)
    last_row = df.iloc[-1]
    last_date = pd.to_datetime(last_row['Date'])
    
    import datetime
    diff_days = (datetime.datetime.now().date() - last_date.date()).days
    if diff_days >= 1:
        st.info(f"💡 장 시작 전입니다. **{last_date.strftime('%Y-%m-%d')} (전일 종가)** 기준으로 분석합니다.")
    else:
        st.caption(f"📅 기준일: **{last_date.strftime('%Y-%m-%d')}** (최신)")
    
    has_market = "Close_mkt" in df.columns
    ma_buy = int(ma_buy)
    ma_sell = int(ma_sell)
    
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

    # [복구] 주말 및 장 시작 전, 다음 거래일 시그널 도출용 '가짜 캔들' 연장 로직
    import datetime
    last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
    today = datetime.datetime.now().date()
    
    if last_date < today:
        dummy_row = df.iloc[-1:].copy() # 지표 계산이 끝난 금요일 캔들을 그대로 복사
        dummy_row['Date'] = pd.to_datetime(today)
        df = pd.concat([df, dummy_row], ignore_index=True)
    
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
            ma_b = float(df["MA_BUY"].iloc[i - int(offset_ma_buy)])
            ma_s = float(df["MA_SELL"].iloc[i - int(offset_ma_sell)])
            trend_ok = True
            t_str = "" # 💡 [추가] 시그널 탭에도 추세텍스트 표기
            
            if (use_trend_in_buy or use_trend_in_sell) and "MA_SHORT" in df.columns:
                s_val = df["MA_SHORT"].iloc[i - int(offset_compare_short)]
                l_val = df["MA_LONG"].iloc[i - int(offset_compare_long)]
                trend_ok = s_val >= l_val
                t_str = f" [추세: 단기{s_val:.2f} {'≥' if trend_ok else '<'} 장기{l_val:.2f}]"

            buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
            
            if sell_operator == "OFF":
                sell_ok = False
                sell_cond_str = "OFF (전략매도 끔)"
            else:
                sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
                sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
                sell_cond_str = f"종가 {cl_s:.2f} {sell_operator} 이평 {ma_s:.2f}"
                if use_trend_in_sell: sell_cond_str += t_str # 💡 [추가]
            
            buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
            cond_str = f"종가 {cl_b:.2f} {buy_operator} 이평 {ma_b:.2f}"
            if use_trend_in_buy: cond_str += t_str # 💡 [추가]

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

def summarize_signal_today(df, p):
    if df is None or df.empty: return {"label": "N/A", "last_buy": "-", "last_sell": "-", "last_hold": "-"}
    try:
        df = df.copy().sort_values("Date").reset_index(drop=True)
        if "Close_sig" in df.columns: 
            df["Close"] = pd.to_numeric(df["Close_sig"], errors="coerce")
        else: 
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            
        if len(df) < 60: return {"label": "데이터부족", "last_buy": "-", "last_sell": "-", "last_hold": "-"}

        idx_now = len(df) - 1
        
        # 파라미터 안전 변환 (문자열 False 방지)
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

        # [복구] 프리셋 탭: 주말 및 장 시작 전, 다음 거래일 시그널 도출용 '가짜 캔들' 연장 로직
        import datetime
        last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
        today = datetime.datetime.now().date()
        
        if last_date < today:
            dummy_row = df.iloc[-1:].copy()
            dummy_row['Date'] = pd.to_datetime(today)
            df = pd.concat([df, dummy_row], ignore_index=True)

        last_buy_date, last_sell_date = "-", "-"
        debug_msg = "" # 💡 [추가] 이유 저장 바구니

        def _check(i, type_):
            nonlocal debug_msg # 💡 [추가]
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
                        res = False
                        if "상단선" in bb_entry: res = cl > df["BB_UP"].iloc[i-off_cl_b]
                        elif "하단선" in bb_entry: res = cl < df["BB_LO"].iloc[i-off_cl_b]
                        else: res = cl > df["BB_MID"].iloc[i-off_cl_b]
                        # 💡 [추가] 볼린저 밴드 관망 이유
                        if i == idx_now and not res:
                            t_info = f", 추세:{'✅' if trend_ok else '❌'}" if use_trend_buy else ""
                            debug_msg = f"(종가 vs 볼린저{t_info})"
                        return res
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
                        # 💡 [추가] 이평선 관망 이유
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
        
        # 💡 [수정] 원본 로직 유지하되 debug_msg만 덧붙임
        label = f"관망 {debug_msg}".strip() if (not is_buy_now and not is_sell_now and debug_msg) else "관망"
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
        
        # [유지] 원본 그대로 - 짝대기 반환 (UI에서 보유 판단)
        return {"label": label, "last_buy": last_buy_date, "last_sell": last_sell_date, "last_hold": "-"}
    except Exception as e: return {"label": f"오류:{e}", "last_buy": "-", "last_sell": "-", "last_hold": "-"}

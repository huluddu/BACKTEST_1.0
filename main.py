import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import google.generativeai as genai
import optuna

# 모듈 불러오기
from modules.utils import load_saved_strategies, save_strategy_to_file, delete_strategy_from_file, parse_choices
from modules.data_loader import get_data, get_fundamental_info
from modules.strategy import prepare_base, check_signal_today, backtest_fast, summarize_signal_today, auto_search_train_test, apply_opt_params, optuna_objective
from modules.llm_advisor import ask_gemini_analysis, ask_gemini_chat, ask_gemini_comprehensive_analysis

st.set_page_config(page_title="QuantLab: Modular Ver.", page_icon="⚡", layout="wide")

# --- [함수 정의] 전략을 한글 문장으로 변환 ---
def translate_strategy_condition(ticker, ma_period, offset_ma, offset_cl, operator):
    ma_time = "현재" if offset_ma == 0 else f"{offset_ma}일 전"
    cl_time = "현재" if offset_cl == 0 else f"{offset_cl}일 전"
    
    op_desc = ""
    if operator == ">": op_desc = "클 때"
    elif operator == "<": op_desc = "작을 때"
    else: op_desc = f"({operator})일 때"

    return f"**{ticker}**의 **{ma_time} {ma_period}일 이평선**이 **{cl_time} 종가**보다 **{op_desc}**"

# --- [함수 수정] 추세/역추세 모두 해석 가능하도록 변경 ---
def translate_trend_condition(ticker, ma_short, off_short, ma_long, off_long, mode="buy"):
    """
    mode="buy": 정배열 (Short > Long)
    mode="sell": 역배열 (Short < Long)
    """
    s_time = "현재" if off_short == 0 else f"{off_short}일 전"
    l_time = "현재" if off_long == 0 else f"{off_long}일 전"
    
    s_desc = f"**{s_time} {ma_short}일 이평선**"
    l_desc = f"**{l_time} {ma_long}일 이평선**"

    if mode == "buy":
        return f"{s_desc}이 {l_desc}보다 **클 때 (정배열)**"
    else:
        return f"{s_desc}이 {l_desc}보다 **작을 때 (역배열/데드크로스)**"

# ==========================================
# 1. 초기 상태 및 프리셋 설정
# ==========================================
def _init_default_state():
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
    defaults = {
        "signal_ticker_input": "SOXL", "trade_ticker_input": "SOXL", "market_ticker_input": "SPY", 
        "buy_operator": ">", "sell_operator": "<", "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "offset_cl_buy": 1, "offset_cl_sell": 1, "offset_ma_buy": 1, "offset_ma_sell": 1,
        "ma_buy": 50, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": False,
        "ma_compare_short": 20, "ma_compare_long": 50, "offset_compare_short": 1, "offset_compare_long": 1,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0, "min_hold_days": 0, "fee_bps": 25, "slip_bps": 1,
        "preset_name": "직접 설정", "gemini_api_key": "", "auto_run_trigger": False,
        "use_rsi_filter": False, "rsi_period": 14, "rsi_min": 30, "rsi_max": 70,
        "use_market_filter": False, "market_ma_period": 200,
        "use_bollinger": False, "bb_period": 20, "bb_std": 2.0,
        "bb_entry_type": "상단선 돌파 (추세)", "bb_exit_type": "중심선(MA) 이탈",
        # [ATR 기능 초기값 추가]
        "use_atr_stop": False, "atr_multiplier": 2.0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

_init_default_state()

# ---------------------------------------------------------
# [복구 완료] 사용자님의 원본 프리셋 데이터 전체
# ---------------------------------------------------------
DEFAULT_PRESETS = {
}

# 로컬 파일(구글 시트 등)에 저장된 전략이 있다면 합치기
try:
    saved_strategies = load_saved_strategies()
    if saved_strategies:
        DEFAULT_PRESETS.update(saved_strategies)
except Exception as e:
    st.toast(f"⚠️ 전략 로드 실패: {e}")

PRESETS = DEFAULT_PRESETS
st.session_state["ALL_PRESETS_DATA"] = PRESETS

def _on_preset_change():
    name = st.session_state["preset_name_selector"]
    st.session_state["preset_name"] = name
    preset = st.session_state.get("ALL_PRESETS_DATA", {}).get(name, {})
    if not preset: return

    for k, v in preset.items():
        key_name = k
        if k == "signal_ticker": key_name = "signal_ticker_input"
        elif k == "trade_ticker": key_name = "trade_ticker_input"
        elif k == "market_ticker": key_name = "market_ticker_input"
        
        if key_name in st.session_state:
            st.session_state[key_name] = v

# ==========================================
# 2. 사이드바 (설정 & 저장)
# ==========================================
with st.sidebar:
    st.header("⚙️ 설정 & Gemini")
    
    # API 키 입력
    api_key_input = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
    if api_key_input: 
        st.session_state["gemini_api_key"] = api_key_input
        try:
            genai.configure(api_key=api_key_input)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.session_state["selected_model_name"] = st.selectbox("🤖 모델 선택", models, index=0)
        except: 
            st.error("모델 로드 실패")
    
    st.divider()

    with st.expander("💾 전략 저장/삭제"):
        with st.form("strategy_save_form", clear_on_submit=False):
            save_name = st.text_input("새 전략 이름 입력")
            submitted = st.form_submit_button("현재 설정 저장하기")
            
            if submitted:
                if save_name:
                    keys_to_save = [
                        "signal_ticker_input", "trade_ticker_input", "market_ticker_input",
                        "buy_operator", "sell_operator", "strategy_behavior",
                        "ma_buy", "ma_sell", 
                        "offset_cl_buy", "offset_cl_sell", "offset_ma_buy", "offset_ma_sell",
                        "use_trend_in_buy", "use_trend_in_sell",
                        "ma_compare_short", "ma_compare_long", "offset_compare_short", "offset_compare_long",
                        "stop_loss_pct", "take_profit_pct", "min_hold_days",
                        "fee_bps", "slip_bps",
                        "use_market_filter", "market_ma_period",
                        "use_bollinger", "bb_period", "bb_std", "bb_entry_type", "bb_exit_type",
                        "use_rsi_filter", "rsi_period", "rsi_max",
                        # [추가됨] ATR 설정 저장
                        "use_atr_stop", "atr_multiplier"
                    ]
                    params = {k: st.session_state.get(k) for k in keys_to_save}
                    save_strategy_to_file(save_name, params)
                    st.session_state["preset_name_selector"] = save_name
                    st.rerun()
                else:
                    st.error("전략 이름을 입력해주세요!")
        
        del_name = st.selectbox("삭제할 전략 선택", list(PRESETS.keys())) if PRESETS else None
        if del_name and st.button("삭제"):
            delete_strategy_from_file(del_name)
            st.session_state["preset_name_selector"] = "직접 설정"
            st.rerun()

    st.divider()
    
    selected_preset = st.selectbox(
        "🎯 프리셋", 
        ["직접 설정"] + list(PRESETS.keys()), 
        key="preset_name_selector", 
        on_change=_on_preset_change
    )

# ==========================================
# 3. 메인 파라미터 입력창 (상단)
# ==========================================
col1, col2, col3 = st.columns(3)
signal_ticker = col1.text_input("시그널 티커", key="signal_ticker_input")
trade_ticker = col2.text_input("매매 티커", key="trade_ticker_input")
market_ticker = col3.text_input("시장 티커 (옵션)", key="market_ticker_input", help="예: SPY")

col4, col5 = st.columns(2)
start_date = col4.date_input("시작일", value=datetime.date(2020, 1, 1),min_value=datetime.date(1980, 1, 1))
end_date = col5.date_input("종료일", value=datetime.date.today())

# --- 사이드바 상세 설정 UI (전체 교체) ---
with st.expander("📈 상세 설정 (Offset, 비용 등)", expanded=False):
    tabs = st.tabs(["📊 이평선 설정", "🚦 시장 필터", "🌊 볼린저 밴드", "🛡️ 리스크/기타"])

    # 1. 이평선 및 추세선 설정
    with tabs[0]:
        st.markdown("#### 📥 매수 조건")
        c1, c2 = st.columns(2)
        c1.number_input("매수 이평 (MA)", key="ma_buy", step=1, min_value=1)
        c2.number_input("매수 이평 Offset", key="offset_ma_buy", step=1)
        c1.number_input("매수 종가 Offset", key="offset_cl_buy", step=1)
        c2.selectbox("매수 부호", [">", "<"], key="buy_operator")
        st.checkbox("매수 추세 필터 (정배열)", key="use_trend_in_buy")

        st.divider()
        st.markdown("#### 📤 매도 조건")
        c3, c4 = st.columns(2)
        c3.number_input("매도 이평 (MA)", key="ma_sell", step=1, min_value=1)
        c4.number_input("매도 이평 Offset", key="offset_ma_sell", step=1)
        c3.number_input("매도 종가 Offset", key="offset_cl_sell", step=1)
        c4.selectbox("매도 부호", ["<", ">", "OFF"], key="sell_operator")
        st.checkbox("매도 역추세 필터 (역배열)", key="use_trend_in_sell")

        st.divider()
        # [복구된 부분] 추세선 설정
        st.markdown("#### 📈 추세선 설정 (Trend Line)")
        st.caption("추세 필터 사용 시 비교할 두 이평선입니다.")
        
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**단기 추세선 (Short)**")
            st.number_input("기간 (Period)", key="ma_compare_short", step=1, min_value=1)
            st.number_input("오프셋 (Offset)", key="offset_compare_short", step=1)
        with t2:
            st.markdown("**장기 추세선 (Long)**")
            st.number_input("기간 (Period)", key="ma_compare_long", step=1, min_value=1)
            st.number_input("오프셋 (Offset)", key="offset_compare_long", step=1)

    # 2. 시장 필터
    with tabs[1]:
        st.markdown("#### 🚦 시장 필터 (Market Filter)")
        st.write("시장 지수(예: SPY)가 이평선 위에 있을 때만 매수합니다.")
        st.checkbox("시장 필터 사용", key="use_market_filter")
        st.number_input("시장 이평선 기간", value=200, step=10, key="market_ma_period")

    # 3. 볼린저 밴드
    with tabs[2]:
        st.markdown("#### 🌊 볼린저 밴드 (Volatility Breakout)")
        st.write("이평선 매매 대신 볼린저 밴드 돌파 전략을 사용합니다.")
        st.checkbox("볼린저 밴드 사용", key="use_bollinger")
        c_b1, c_b2 = st.columns(2)
        c_b1.number_input("밴드 기간", value=20, key="bb_period")
        c_b2.number_input("밴드 승수 (Std Dev)", value=2.0, step=0.1, key="bb_std")
        st.selectbox("매수 기준", ["상단선 돌파 (추세)", "하단선 이탈 (역추세)", "중심선 돌파"], key="bb_entry_type")
        st.selectbox("매도 기준", ["중심선(MA) 이탈", "상단선 복귀", "하단선 이탈"], key="bb_exit_type")

    # 4. 리스크 및 기타
    with tabs[3]:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### 🛡️ 리스크")
            st.checkbox("ATR(변동성) 손절 사용", key="use_atr_stop")
            if st.session_state.use_atr_stop:
                st.number_input("ATR 배수", value=2.0, step=0.1, key="atr_multiplier")
                st.caption("손절가 = 진입가 - (ATR x 배수)")
                stop_loss_pct = 0.0
            else:
                st.number_input("고정 손절 (%)", step=0.5, key="stop_loss_pct")
            
            st.number_input("익절 (%)", step=0.5, key="take_profit_pct")
            st.number_input("최소 보유일", step=1, key="min_hold_days")
        with c6:
            st.markdown("#### ⚙️ 기타")
            st.selectbox("행동 패턴", ["1. 포지션 없으면 매수 / 보유 중이면 매도", "2. 매수 우선"], key="strategy_behavior")
            st.number_input("수수료 (bps)", value=25, step=1, key="fee_bps")
            st.number_input("슬리피지 (bps)", value=5, step=1, key="slip_bps")
            
        st.divider()
        st.markdown("#### 🔮 보조지표")
        c_r1, c_r2 = st.columns(2)
        c_r1.number_input("RSI 기간", 14, step=1, key="rsi_period")
        st.checkbox("RSI 필터 적용", key="use_rsi_filter")
        if st.session_state.use_rsi_filter:
            c_r2.number_input("RSI 과매수 기준", 70, key="rsi_max")

# ==========================================
# 4. 기능 탭 (기업정보, 시그널, 프리셋, 백테스트, 실험실)
# ==========================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏢 기업 정보", "🎯 시그널", "📚 PRESETS", "🧪 백테스트", "🧬 실험실", "🧮 손절 계산기", "📊 펀더멘털", "AI 삐빅"])

with tab0:
    st.markdown("### 🏢 기업 기본 정보 (Fundamental)")
    if trade_ticker:
        fd = get_fundamental_info(trade_ticker)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("기업명", fd["Name"])
        c2.metric("섹터", fd["Sector"])
        c3.metric("시가총액", f"{fd['MarketCap']:,}")
        c4.metric("Beta (변동성)", f"{fd['Beta']:.2f}")
        
        st.divider()
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PER (주가수익비율)", f"{fd['PER']:.2f}" if fd['PER'] else "N/A")
        c6.metric("PBR (주가순자산비율)", f"{fd['PBR']:.2f}" if fd['PBR'] else "N/A")
        c7.metric("ROE (자기자본이익률)", f"{fd['ROE'] * 100:.2f}%" if fd['ROE'] else "N/A")
        c8.metric("당기순이익", f"{fd['NetIncome']:,}")

        st.info(f"ℹ️ **기업 개요**: {fd['Description']}")
    else:
        st.warning("티커를 입력해주세요.")

with tab1:
    if st.button("📌 오늘의 매매 시그널 확인", type="primary", use_container_width=True):
        base, x_sig, x_trd, ma_dict, x_mkt, ma_mkt_arr = prepare_base(
            signal_ticker, trade_ticker, market_ticker, start_date, end_date, 
            [st.session_state.ma_buy, st.session_state.ma_sell, st.session_state.ma_compare_short, st.session_state.ma_compare_long], 
            st.session_state.market_ma_period
        )
        if base is not None:
             check_signal_today(base, st.session_state.ma_buy, st.session_state.offset_ma_buy, st.session_state.ma_sell, st.session_state.offset_ma_sell, st.session_state.offset_cl_buy, st.session_state.offset_cl_sell, st.session_state.ma_compare_short, st.session_state.ma_compare_long, st.session_state.offset_compare_short, st.session_state.offset_compare_long, st.session_state.buy_operator, st.session_state.sell_operator, st.session_state.use_trend_in_buy, st.session_state.use_trend_in_sell,
                                st.session_state.use_market_filter, market_ticker, st.session_state.market_ma_period, 
                                st.session_state.use_bollinger, st.session_state.bb_period, st.session_state.bb_std, st.session_state.bb_entry_type, st.session_state.bb_exit_type)
        else: st.error("데이터 로딩 실패")

# --- tab2 전체 교체 ---
# --- Tab 2: 프리셋 전체 분석 ---
with tab2:
    st.markdown("### 📚 전략 일괄 진단 & 기간별 스트레스 테스트")
    
    # 백테스트는 항상 실행 (화면엔 안 보임)
    run_full_backtest = True 
    
    # 탭 분리
    sub_tab1, sub_tab2 = st.tabs(["🚀 현재 설정 분석 (보유종목 확인)", "🗓️ 5/10/15/20년 상세 검증"])

    # ---------------------------------------------------------
    # 1. 현재 설정 기준 분석
    # ---------------------------------------------------------
    with sub_tab1:
        st.info(f"사이드바에 설정된 기간 (**{start_date} ~ {end_date}**)을 기준으로 현재 상태를 진단합니다.")
        
        if st.button("🚀 분석 시작 (현재 설정)", type="primary"):
            rows = []
            progress_text = "전략 분석 중..."
            my_bar = st.progress(0, text=progress_text)
            total_presets = len(PRESETS)
            
            for i, (name, p) in enumerate(PRESETS.items()):
                my_bar.progress(int((i / total_presets) * 100), text=f"분석 중: {name}")
                
                s_ticker = p.get("signal_ticker", p.get("signal_ticker_input", "SOXL"))
                t_ticker = p.get("trade_ticker", p.get("trade_ticker_input", "SOXL"))
                m_ticker = p.get("market_ticker", p.get("market_ticker_input", "SPY"))
                
                ma_pool = [
                    int(p.get("ma_buy", 50)), int(p.get("ma_sell", 10)),
                    int(p.get("ma_compare_short", 0) or 0), int(p.get("ma_compare_long", 0) or 0)
                ]
                
                base, x_sig, x_trd, ma_dict, x_mkt, ma_mkt_arr = prepare_base(
                    s_ticker, t_ticker, m_ticker, start_date, end_date, ma_pool, 
                    int(p.get("market_ma_period", 200))
                )
                
                if base is not None and not base.empty:
                    # 시그널 요약
                    sig_res = summarize_signal_today(get_data(s_ticker, start_date, end_date), p)
                    
                    row_data = {
                        "전략명": name, 
                        "티커": t_ticker, # [수정] s_ticker -> t_ticker (매매 티커 기준)
                        "현재상태": sig_res["label"], 
                        "최근매수": sig_res["last_buy"],
                        "보유여부": "❓ 미확인"
                    }

                    # 백테스트 실행
                    bt_res = backtest_fast(
                        base, x_sig, x_trd, ma_dict,
                        int(p.get("ma_buy", 50)), int(p.get("offset_ma_buy", 0)),
                        int(p.get("ma_sell", 10)), int(p.get("offset_ma_sell", 0)),
                        int(p.get("offset_cl_buy", 0)), int(p.get("offset_cl_sell", 0)),
                        int(p.get("ma_compare_short", 0) or 0), int(p.get("ma_compare_long", 0) or 0),
                        int(p.get("offset_compare_short", 0)), int(p.get("offset_compare_long", 0)),
                        5000000, 
                        float(p.get("stop_loss_pct", 0.0)), float(p.get("take_profit_pct", 0.0)),
                        str(p.get("strategy_behavior", "1")), int(p.get("min_hold_days", 0)),
                        float(p.get("fee_bps", 25)), float(p.get("slip_bps", 1)),
                        bool(p.get("use_trend_in_buy", True)), bool(p.get("use_trend_in_sell", False)),
                        str(p.get("buy_operator", ">")), str(p.get("sell_operator", "<")),
                        use_rsi_filter=bool(p.get("use_rsi_filter", False)),
                        rsi_period=int(p.get("rsi_period", 14)), rsi_min=30, rsi_max=int(p.get("rsi_max", 70)),
                        use_market_filter=bool(p.get("use_market_filter", False)),
                        x_mkt=x_mkt, ma_mkt_arr=ma_mkt_arr,
                        use_bollinger=bool(p.get("use_bollinger", False)),
                        bb_period=int(p.get("bb_period", 20)), bb_std=float(p.get("bb_std", 2.0)),
                        bb_entry_type=str(p.get("bb_entry_type", "")), bb_exit_type=str(p.get("bb_exit_type", "")),
                        use_atr_stop=bool(p.get("use_atr_stop", False)),
                        atr_multiplier=float(p.get("atr_multiplier", 2.0))
                    )
                    
                    # 보유 여부, 날짜, 매수가 표시 로직
                    hold_status = "⚪ 미보유"
                    buy_price_display = "-" # 매수가 초기값
                    trades = bt_res.get('매매 로그', [])
                    
                    if trades:
                        last_trade = trades[-1]
                        if last_trade.get('신호') == 'BUY':
                            buy_date = last_trade.get('날짜')
                            buy_price = last_trade.get('체결가', 0) # 👈 체결가 가져오기
                            
                            if isinstance(buy_date, pd.Timestamp):
                                buy_date_str = buy_date.strftime("%Y-%m-%d")
                            else:
                                buy_date_str = str(buy_date)[:10]
                                
                            hold_status = f"🟢 보유중 ({buy_date_str})"
                            buy_price_display = f"${buy_price:,.2f}" # 👈 매수가 포맷팅
                    
                    row_data.update({
                        "보유여부": hold_status,
                        "매수가": buy_price_display, # 👈 결과 행에 추가
                        "총 수익률(%)": f"{bt_res.get('수익률 (%)', 0)}%",
                        "MDD(%)": f"{bt_res.get('MDD (%)', 0)}%",
                        "승률(%)": f"{bt_res.get('승률 (%)', 0)}%",
                        "매매횟수": bt_res.get('총 매매 횟수', 0)
                    })
                    
                    rows.append(row_data)
                else:
                    rows.append({"전략명": name, "티커": t_ticker, "보유여부": "❌ 에러", "현재상태": "데이터오류"})

            my_bar.empty()
            
            if rows:
                df_result = pd.DataFrame(rows)
                
                if "총 수익률(%)" in df_result.columns:
                    try:
                        df_result["sort"] = df_result["총 수익률(%)"].str.replace("%", "").astype(float)
                        df_result = df_result.sort_values("sort", ascending=False).drop(columns=["sort"])
                    except: pass
                
                st.success("✅ 분석 완료!")
                
                # 정렬 및 컬럼 순서 (매수가를 보유여부 옆에 배치)
                cols_order = ["전략명", "티커", "보유여부", "매수가", "현재상태", "총 수익률(%)", "MDD(%)", "승률(%)", "매매횟수"]
                final_cols = [c for c in cols_order if c in df_result.columns]
                
                st.dataframe(
                    df_result[final_cols], 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "전략명": st.column_config.TextColumn("전략", width="medium"),
                        "티커": st.column_config.TextColumn("매매 종목", width="small"),
                        "보유여부": st.column_config.TextColumn("보유 상태", width="medium"),
                        "매수가": st.column_config.TextColumn("진입 가격", width="small", help="현재 보유 중인 포지션의 매수 단가"), # 👈 설정 추가
                        "현재상태": st.column_config.TextColumn("오늘 시그널"),
                    }
                )
            else:
                st.warning("분석할 프리셋이 없습니다.")

    # ---------------------------------------------------------
    # 2. 5/10/15/20년 멀티 백테스트 (매매 티커 기준)
    # ---------------------------------------------------------
    with sub_tab2:
        st.write("##### ⏳ 과거 4개 구간(5/10/15/20년) 상세 검증")
        st.caption("대분류(지표) 하위에 기간별 데이터를 보여줍니다.")
        
        if st.button("🗓️ 역사적 구간 분석 시작", type="primary"):
            periods = [5, 10, 15, 20]
            data_list = []
            
            total_steps = len(PRESETS) * len(periods)
            p_bar = st.progress(0, text="멀티 백테스트 준비 중...")
            step_count = 0
            today = datetime.date.today()
            
            for name, p in PRESETS.items():
                s_ticker = p.get("signal_ticker", p.get("signal_ticker_input", "SOXL"))
                t_ticker = p.get("trade_ticker", p.get("trade_ticker_input", "SOXL"))
                m_ticker = p.get("market_ticker", p.get("market_ticker_input", "SPY"))
                
                # 전략 식별자 (매매 티커 표시)
                # [수정] s_ticker -> t_ticker
                strategy_idx = f"{name} ({t_ticker})"
                row_data = {}
                
                for yr in periods:
                    step_count += 1
                    p_bar.progress(int((step_count / total_steps) * 100), text=f"[{name}] {yr}년 데이터 분석 중...")
                    start_d = today - datetime.timedelta(days=365 * yr)
                    
                    ma_pool = [
                        int(p.get("ma_buy", 50)), int(p.get("ma_sell", 10)),
                        int(p.get("ma_compare_short", 0) or 0), int(p.get("ma_compare_long", 0) or 0)
                    ]
                    
                    try:
                        base, x_sig, x_trd, ma_dict, x_mkt, ma_mkt_arr = prepare_base(
                            s_ticker, t_ticker, m_ticker, start_d, today, ma_pool, 
                            int(p.get("market_ma_period", 200))
                        )
                        
                        if base is not None and not base.empty:
                            res = backtest_fast(
                                base, x_sig, x_trd, ma_dict,
                                int(p.get("ma_buy", 50)), int(p.get("offset_ma_buy", 0)),
                                int(p.get("ma_sell", 10)), int(p.get("offset_ma_sell", 0)),
                                int(p.get("offset_cl_buy", 0)), int(p.get("offset_cl_sell", 0)),
                                int(p.get("ma_compare_short", 0) or 0), int(p.get("ma_compare_long", 0) or 0),
                                int(p.get("offset_compare_short", 0)), int(p.get("offset_compare_long", 0)),
                                5000000, 
                                float(p.get("stop_loss_pct", 0.0)), float(p.get("take_profit_pct", 0.0)),
                                str(p.get("strategy_behavior", "1")), int(p.get("min_hold_days", 0)),
                                float(p.get("fee_bps", 25)), float(p.get("slip_bps", 1)),
                                bool(p.get("use_trend_in_buy", True)), bool(p.get("use_trend_in_sell", False)),
                                str(p.get("buy_operator", ">")), str(p.get("sell_operator", "<")),
                                use_rsi_filter=bool(p.get("use_rsi_filter", False)),
                                rsi_period=int(p.get("rsi_period", 14)), rsi_min=30, rsi_max=int(p.get("rsi_max", 70)),
                                use_market_filter=bool(p.get("use_market_filter", False)),
                                x_mkt=x_mkt, ma_mkt_arr=ma_mkt_arr,
                                use_bollinger=bool(p.get("use_bollinger", False)),
                                bb_period=int(p.get("bb_period", 20)), bb_std=float(p.get("bb_std", 2.0)),
                                bb_entry_type=str(p.get("bb_entry_type", "")), bb_exit_type=str(p.get("bb_exit_type", "")),
                                use_atr_stop=bool(p.get("use_atr_stop", False)),
                                atr_multiplier=float(p.get("atr_multiplier", 2.0))
                            )
                            
                            real_start = base['Date'].iloc[0].date()
                            years_avail = round((today - real_start).days / 365, 1)
                            suffix = f" ({years_avail}y)" if years_avail < (yr - 0.5) else ""
                            
                            row_data[('수익률', f"{yr}년")] = f"{res.get('수익률 (%)', 0)}%{suffix}"
                            row_data[('MDD', f"{yr}년")] = f"{res.get('MDD (%)', 0)}%"
                            row_data[('승률', f"{yr}년")] = f"{res.get('승률 (%)', 0)}%"
                            row_data[('매매횟수', f"{yr}년")] = f"{res.get('총 매매 횟수', 0)}회"
                        else:
                            for cat in ['수익률', 'MDD', '승률', '매매횟수']: row_data[(cat, f"{yr}년")] = "-"
                    except:
                        for cat in ['수익률', 'MDD', '승률', '매매횟수']: row_data[(cat, f"{yr}년")] = "Err"

                row_data[('전략', '이름')] = strategy_idx
                data_list.append(row_data)
            
            p_bar.empty()
            st.success("✅ 통합 분석 완료!")
            
            if data_list:
                df_raw = pd.DataFrame(data_list)
                if ('전략', '이름') in df_raw.columns:
                    df_raw.set_index(('전략', '이름'), inplace=True)
                    df_raw.index.name = "전략명 (매매종목)"
                
                desired_cols = []
                for cat in ['수익률', 'MDD', '승률', '매매횟수']:
                    for yr in periods: desired_cols.append((cat, f"{yr}년"))
                
                final_cols = [c for c in desired_cols if c in df_raw.columns]
                st.dataframe(df_raw[final_cols], use_container_width=True)
                
with tab3:
    if st.button("✅ 백테스트 실행 (종가매매)", type="primary", use_container_width=True):
        
        p_ma_buy = int(st.session_state.ma_buy)
        p_ma_sell = int(st.session_state.ma_sell)
        p_ma_compare_short = int(st.session_state.ma_compare_short) if st.session_state.ma_compare_short else 0
        p_ma_compare_long = int(st.session_state.ma_compare_long) if st.session_state.ma_compare_long else 0
        
        ma_pool = [p_ma_buy, p_ma_sell, p_ma_compare_short, p_ma_compare_long]
        base, x_sig, x_trd, ma_dict, x_mkt, ma_mkt_arr = prepare_base(signal_ticker, trade_ticker, market_ticker, start_date, end_date, ma_pool, st.session_state.market_ma_period)
        
        if base is not None:
            with st.spinner("과거 데이터를 한 땀 한 땀 분석 중..."):
                p_use_rsi = st.session_state.get("use_rsi_filter", False)
                p_rsi_period = st.session_state.get("rsi_period", 14)
                p_rsi_max = st.session_state.get("rsi_max", 70)

                res = backtest_fast(base, x_sig, x_trd, ma_dict, p_ma_buy, st.session_state.offset_ma_buy, p_ma_sell, st.session_state.offset_ma_sell, st.session_state.offset_cl_buy, st.session_state.offset_cl_sell, p_ma_compare_short, p_ma_compare_long, st.session_state.offset_compare_short, st.session_state.offset_compare_long, 5000000, st.session_state.stop_loss_pct, st.session_state.take_profit_pct, st.session_state.strategy_behavior, st.session_state.min_hold_days, st.session_state.fee_bps, st.session_state.slip_bps, st.session_state.use_trend_in_buy, st.session_state.use_trend_in_sell, st.session_state.buy_operator, st.session_state.sell_operator, 
                                use_rsi_filter=p_use_rsi, rsi_period=p_rsi_period, rsi_min=30, rsi_max=p_rsi_max,
                                use_market_filter=st.session_state.use_market_filter, x_mkt=x_mkt, ma_mkt_arr=ma_mkt_arr,
                                use_bollinger=st.session_state.use_bollinger, bb_period=st.session_state.bb_period, bb_std=st.session_state.bb_std, 
                                bb_entry_type=st.session_state.bb_entry_type, bb_exit_type=st.session_state.bb_exit_type,
                                # [추가됨] ATR 파라미터 전달
                                use_atr_stop=st.session_state.get("use_atr_stop", False),
                                atr_multiplier=st.session_state.get("atr_multiplier", 2.0))
            st.session_state["bt_result"] = res
            if "ai_analysis" in st.session_state: del st.session_state["ai_analysis"]
            st.rerun()
        else: st.error("데이터 로딩 실패")

    if "bt_result" in st.session_state:
        res = st.session_state["bt_result"]

        # =========================================================
        # [전략 해석 표시]
        # =========================================================
        st.divider()
        st.markdown("### 📖 전략 해석")

        # (1) 매수 조건
        buy_main = translate_strategy_condition(
            signal_ticker, 
            st.session_state.ma_buy, st.session_state.offset_ma_buy, st.session_state.offset_cl_buy, st.session_state.buy_operator
        )
        
        # (2) 매수 추세 필터 (정배열)
        buy_trend = ""
        if st.session_state.use_trend_in_buy:
            t_txt = translate_trend_condition(
                signal_ticker,
                st.session_state.ma_compare_short, st.session_state.offset_compare_short,
                st.session_state.ma_compare_long, st.session_state.offset_compare_long,
                mode="buy"
            )
            buy_trend = f"\n  - ➕ **추세 필터:** {t_txt}"

        # (3) 매도 조건
        sell_main = translate_strategy_condition(
            signal_ticker, 
            st.session_state.ma_sell, st.session_state.offset_ma_sell, st.session_state.offset_cl_sell, st.session_state.sell_operator
        )

        # (4) 매도 역추세 필터 (역배열) - [수정됨] 상세 표시
        sell_trend = ""
        if st.session_state.use_trend_in_sell:
            t_txt = translate_trend_condition(
                signal_ticker,
                st.session_state.ma_compare_short, st.session_state.offset_compare_short,
                st.session_state.ma_compare_long, st.session_state.offset_compare_long,
                mode="sell"
            )
            sell_trend = f"\n  - ➕ **역추세 필터:** {t_txt}"

        # 화면 출력
        st.info(f"🔵 **매수 진입:** {buy_main}{buy_trend}\n\n🔴 **매도 청산:** {sell_main}{sell_trend}")
        st.divider()
        # =========================================================        
        
        if res:
            # ---------------------------------------
            # [NEW] B&H(단순보유) 성과 계산 로직 추가
            # ---------------------------------------
            bh_return = 0.0
            bh_mdd = 0.0
            
            df_log = pd.DataFrame(res['매매 로그'])
            
            if not df_log.empty:
                # 1. B&H 수익률
                first_price = df_log['종가'].iloc[0]
                last_price = df_log['종가'].iloc[-1]
                bh_return = ((last_price - first_price) / first_price) * 100
                
                # 2. B&H MDD
                # (가격 흐름 자체가 자산 곡선이 됨)
                price_series = df_log['종가']
                running_max = price_series.cummax()
                drawdown = (price_series - running_max) / running_max * 100
                bh_mdd = drawdown.min()

            # ---------------------------------------
            # [NEW] 메트릭 표시 (전략 vs B&H 비교)
            # ---------------------------------------
            k1, k2, k3, k4 = st.columns(4)
            
            # 수익률: 전략값 보여주고, 작은 글씨(delta)로 B&H 수익률 표시
            k1.metric(
                "총 수익률", 
                f"{res['수익률 (%)']}%", 
                f"B&H: {bh_return:.1f}%", 
                delta_color="off" # 색상 끄기 (단순 비교용)
            )
            
            # MDD: 전략값 보여주고, 작은 글씨로 B&H MDD 표시
            k2.metric(
                "MDD (최대낙폭)", 
                f"{res['MDD (%)']}%", 
                f"B&H: {bh_mdd:.1f}%",
                delta_color="inverse" # MDD는 음수니까 색상 반전 (빨간색이 나쁨)
            )
            
            k3.metric("승률", f"{res['승률 (%)']}%")
            k4.metric("Profit Factor", res['Profit Factor'])
            
            # ---------------------------------------
            # (아래는 기존 차트 그리기 코드 그대로 유지)
            # ---------------------------------------
            if not df_log.empty:
                initial_price = df_log['종가'].iloc[0]
                benchmark = (df_log['종가'] / initial_price) * 5000000
                drawdown = (df_log['자산'] - df_log['자산'].cummax()) / df_log['자산'].cummax() * 100

                chart_data = res.get("차트데이터", {})
                base_df = chart_data.get("base")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25], 
                                    subplot_titles=("주가 & 매매타점 (Candle + MA)", "내 자산 vs 보유 전략 (Equity)", "MDD (%)"))

                if base_df is not None:
                    fig.add_trace(go.Candlestick(x=base_df['Date'], open=base_df['Open_trd'], high=base_df['High_trd'], low=base_df['Low_trd'], close=base_df['Close_trd'], name='가격(Signal)'), row=1, col=1)
                    
                    if st.session_state.use_bollinger and chart_data.get("bb_up") is not None:
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['bb_up'], name='BB 상단', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['bb_lo'], name='BB 하단', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'), row=1, col=1)
                    else:
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['ma_buy_arr'], name='매수 기준선(MA)', line=dict(color='orange', width=1)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['ma_sell_arr'], name='매도 기준선(MA)', line=dict(color='blue', width=1, dash='dot')), row=1, col=1)

                buys = df_log[df_log['신호']=='BUY']
                sells_reg = df_log[(df_log['신호']=='SELL') & (df_log['손절발동']==False) & (df_log['익절발동']==False)]
                sl = df_log[df_log['손절발동']==True]
                tp = df_log[df_log['익절발동']==True]

                fig.add_trace(go.Scatter(x=buys['날짜'], y=buys['종가'], mode='markers', marker=dict(color='#00FF00', symbol='triangle-up', size=12), name='매수 체결'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells_reg['날짜'], y=sells_reg['종가'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=12), name='매도 체결'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sl['날짜'], y=sl['종가'], mode='markers', marker=dict(color='purple', symbol='x', size=12), name='손절'), row=1, col=1)
                fig.add_trace(go.Scatter(x=tp['날짜'], y=tp['종가'], mode='markers', marker=dict(color='gold', symbol='star', size=15), name='익절'), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_log['날짜'], y=df_log['자산'], name='내 전략 자산', line=dict(color='#00F0FF', width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_log['날짜'], y=benchmark, name='단순 보유(Buy&Hold)', line=dict(color='gray', dash='dot')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_log['날짜'], y=drawdown, name='MDD', line=dict(color='#FF4B4B', width=1), fill='tozeroy'), row=3, col=1)

                fig.update_layout(height=900, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📅 월별 수익률 Heatmap")
                df_log['Year'] = df_log['날짜'].dt.year
                df_log['Month'] = df_log['날짜'].dt.month
                df_log['Returns'] = df_log['자산'].pct_change()
                monthly_ret = df_log.groupby(['Year', 'Month'])['Returns'].apply(lambda x: (x + 1).prod() - 1).reset_index()
                pivot_ret = monthly_ret.pivot(index='Year', columns='Month', values='Returns')
                fig_heat = go.Figure(data=go.Heatmap(z=pivot_ret.values * 100, x=pivot_ret.columns, y=pivot_ret.index, colorscale='RdBu', zmid=0, texttemplate="%{z:.1f}%"))
                fig_heat.update_layout(height=400, margin=dict(t=30, b=30))
                st.plotly_chart(fig_heat, use_container_width=True)

                st.divider()
                st.markdown("### 🤖 제미니 퀀트 컨설턴트")
                chat_container = st.container(height=300)
                for msg in st.session_state["chat_history"]:
                    with chat_container.chat_message(msg["role"]): st.write(msg["content"])

                if prompt := st.chat_input("전략에 대해 질문하세요!"):
                    st.session_state["chat_history"].append({"role": "user", "content": prompt})
                    with chat_container.chat_message("user"): st.write(prompt)
                    with chat_container.chat_message("assistant"):
                        current_p = f"매수:{st.session_state.ma_buy}MA, 매도:{st.session_state.ma_sell}MA, 손절:{st.session_state.stop_loss_pct}%"
                        response = ask_gemini_chat(prompt, res, current_p, trade_ticker, st.session_state["gemini_api_key"], st.session_state.get("selected_model_name"))
                        st.write(response)
                        st.session_state["chat_history"].append({"role": "assistant", "content": response})

                st.markdown("### 💾 결과 저장")
                csv = df_log.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 매매 로그 다운로드 (CSV)", data=csv, file_name=f'backtest_log_{trade_ticker}_{datetime.date.today()}.csv', mime='text/csv')

                st.divider()
                if st.button("✨ AI에게 분석 및 개선점 물어보기", type="primary"):
                    fd = get_fundamental_info(trade_ticker)
                    sl_txt = f"{st.session_state.stop_loss_pct}%" if st.session_state.stop_loss_pct > 0 else "미설정"
                    tp_txt = f"{st.session_state.take_profit_pct}%" if st.session_state.take_profit_pct > 0 else "미설정"
                    current_params = f"매수: {st.session_state.ma_buy}일 이평, 매도: {st.session_state.ma_sell}일 이평, 손절: {sl_txt}, 익절: {tp_txt}"
                    anl = ask_gemini_comprehensive_analysis(res, fd, current_params, trade_ticker, st.session_state.get("gemini_api_key"), st.session_state.get("selected_model_name", "gemini-1.5-flash"))
                    st.session_state["ai_analysis"] = anl       
                
                if "ai_analysis" in st.session_state:
                    st.info(st.session_state["ai_analysis"])
                
                with st.expander("📝 상세 로그 보기"):
                    st.dataframe(df_log, use_container_width=True)
        else:
            st.warning("⚠️ 매매 신호가 발생하지 않았습니다.")

with tab4:
    st.markdown("### 🧬 전략 파라미터 자동 최적화 (Grid Search)")
    st.caption("여러 설정을 자동으로 돌려보고 가장 좋은 수익률을 찾아냅니다.")
    
    with st.expander("🔎 필터 및 정렬 설정", expanded=True):
        c1, c2 = st.columns(2)
        sort_metric = c1.selectbox("정렬 기준", ["Full_수익률(%)", "Test_수익률(%)", "Full_MDD(%)", "Full_승률(%)"])
        top_n = c2.slider("표시할 상위 개수", 1, 50, 10)
        
        c3, c4 = st.columns(2)
        min_trades = c3.number_input("최소 매매 횟수", 0, 100, 5)
        min_win = c4.number_input("최소 승률 (%)", 0.0, 100.0, 50.0)
        
        c5, c6 = st.columns(2)
        min_train_ret = c5.number_input("최소 Train 수익률 (%)", -100.0, 1000.0, 0.0)
        min_test_ret = c6.number_input("최소 Test 수익률 (%)", -100.0, 1000.0, 0.0)
        
        limit_mdd = st.number_input("최대 낙폭(MDD) 한계 (%, 절대값)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    colL, colR = st.columns(2)
    with colL:
        st.markdown("#### 1. 매수/매도 조건")
        cand_off_cl_buy = st.text_input("매수 종가 Offset", "1, 5, 10, 20, 50")
        cand_buy_op = st.text_input("매수 부호", "<,>")
        cand_off_ma_buy = st.text_input("매수 이평 Offset", "1, 5, 10, 20, 50")
        cand_ma_buy = st.text_input("매수 이평 (MA Buy)", "1, 5, 10, 20, 50, 60, 120")
        
        st.divider()
        cand_off_cl_sell = st.text_input("매도 종가 Offset", "1, 5, 10, 20, 50")
        cand_sell_op = st.text_input("매도 부호", "<,>,OFF")
        cand_off_ma_sell = st.text_input("매도 이평 Offset", "1, 5, 10, 20, 50")
        cand_ma_sell = st.text_input("매도 이평 (MA Sell)", "1, 5, 10, 20, 50, 60, 120")

    with colR:
        st.markdown("#### 2. 추세 & 리스크")
        cand_use_tr_buy = st.text_input("매수 추세필터 (True, False)", "True, False")
        cand_use_tr_sell = st.text_input("매도 역추세필터", "True")
        
        cand_ma_s = st.text_input("추세 Short 후보", "1, 5, 10, 20, 50, 60, 120")
        cand_ma_l = st.text_input("추세 Long 후보", "1, 5, 10, 20, 50, 60, 120")
        cand_off_s = st.text_input("추세 Short Offset", "1, 5, 10, 20, 50")
        cand_off_l = st.text_input("추세 Long Offset", "1, 5, 10, 20, 50")
        
        st.divider()
        cand_stop = st.text_input("손절(%) 후보 (0=미사용)", "15, 25, 35")
        cand_take = st.text_input("익절(%) 후보", "0, 15, 25, 35")
        
        # [추가됨] ATR 실험 설정
        st.markdown("##### 📉 ATR 손절 실험")
        cand_use_atr = st.text_input("ATR 사용 여부", "False")
        cand_atr_mult = st.text_input("ATR 배수 후보", "2")

    n_trials = st.number_input("시도 횟수", 10, 1000, 100)
    split_ratio = st.slider("Train 비율", 0.0, 1.0, 0.5)
    
    if st.button("🚀 최적 조합 찾기 시작"):
        choices = {
            "ma_buy": parse_choices(cand_ma_buy, "int"), "offset_ma_buy": parse_choices(cand_off_ma_buy, "int"),
            "offset_cl_buy": parse_choices(cand_off_cl_buy, "int"), "buy_operator": parse_choices(cand_buy_op, "str"),
            "ma_sell": parse_choices(cand_ma_sell, "int"), "offset_ma_sell": parse_choices(cand_off_ma_sell, "int"),
            "offset_cl_sell": parse_choices(cand_off_cl_sell, "int"), "sell_operator": parse_choices(cand_sell_op, "str"),
            "use_trend_in_buy": parse_choices(cand_use_tr_buy, "bool"), "use_trend_in_sell": parse_choices(cand_use_tr_sell, "bool"),
            "ma_compare_short": parse_choices(cand_ma_s, "int"), "ma_compare_long": parse_choices(cand_ma_l, "int"),
            "offset_compare_short": parse_choices(cand_off_s, "int"), "offset_compare_long": parse_choices(cand_off_l, "int"),
            "stop_loss_pct": parse_choices(cand_stop, "float"), "take_profit_pct": parse_choices(cand_take, "float"),
            # [추가됨] ATR 실험
            "use_atr_stop": parse_choices(cand_use_atr, "bool"),
            "atr_multiplier": parse_choices(cand_atr_mult, "float")
        }
        
        constraints = {
            "min_trades": min_trades, "min_winrate": min_win, "limit_mdd": limit_mdd,
            "min_train_ret": min_train_ret, "min_test_ret": min_test_ret
        }
        
        with st.spinner("AI가 최적의 파라미터를 탐색 중입니다..."):
            df_opt = auto_search_train_test(
                signal_ticker, trade_ticker, start_date, end_date, split_ratio, choices, 
                n_trials=int(n_trials), initial_cash=5000000, 
                fee_bps=st.session_state.fee_bps, slip_bps=st.session_state.slip_bps, strategy_behavior=st.session_state.strategy_behavior, min_hold_days=st.session_state.min_hold_days,
                constraints=constraints
            )
            
            if not df_opt.empty:
                for col in df_opt.columns:
                    try:
                        df_opt[col] = pd.to_numeric(df_opt[col])
                    except (ValueError, TypeError):
                        pass  # 숫자로 변환할 수 없는 컬럼(문자열 등)은 에러를 무시하고 그대로 둠
                df_opt = df_opt.round(2)

                st.session_state['opt_results'] = df_opt 
                st.session_state['sort_metric'] = sort_metric
            else:
                st.warning("조건을 만족하는 결과가 없습니다.")

    if 'opt_results' in st.session_state:
        df_show = st.session_state['opt_results'].sort_values(st.session_state['sort_metric'], ascending=False).head(top_n)
        st.markdown("#### 🏆 상위 결과 (적용 버튼을 누르면 즉시 백테스트 실행)")
        for i, row in df_show.iterrows():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.dataframe(pd.DataFrame([row]), hide_index=True, use_container_width=True)
            with c2:
                if st.button(f"🥇 적용하기 #{i}", key=f"apply_{i}", on_click=apply_opt_params, args=(row,)):
                    st.rerun()


with tab5:
    st.markdown("### 🧮 매매 계획 계산기 (손절 & 익절)")
    st.caption("진입 정보를 입력하면, ATR(변동성)과 고정 비율(%) 기준의 목표가를 비교해줍니다.")

    # 1. 기본 정보 입력
    c1, c2, c3 = st.columns(3)
    calc_ticker = c1.text_input("종목 티커", value="SOXL", key="calc_ticker")
    calc_date = c2.date_input("매수(진입) 날짜", value=datetime.date.today(), key="calc_date")
    calc_price = c3.number_input("매수 가격 ($)", value=0.0, step=0.1, format="%.2f", key="calc_price")
    
    st.divider()
    
    # 2. 설정 입력 (ATR vs 고정%)
    col_input_l, col_input_r = st.columns(2)
    
    with col_input_l:
        st.info("🌊 ATR (변동성) 기준 설정")
        c_l1, c_l2 = st.columns(2)
        calc_atr_sl = c_l1.number_input("손절 배수 (SL)", value=2.0, step=0.5, help="보통 2~3배를 사용합니다.")
        calc_atr_tp = c_l2.number_input("익절 배수 (TP)", value=4.0, step=0.5, help="손절 배수의 2배 정도가 이상적입니다.")
    
    with col_input_r:
        st.success("🛑 고정 비율 (%) 기준 설정")
        c_r1, c_r2 = st.columns(2)
        calc_pct_sl = c_r1.number_input("손절 비율 (%)", value=5.0, step=1.0)
        calc_pct_tp = c_r2.number_input("익절 비율 (%)", value=10.0, step=1.0)
    
    # 3. 계산 버튼 및 로직
    if st.button("🧮 손익 계산하기", type="primary", use_container_width=True):
        if not calc_ticker or calc_price <= 0:
            st.error("티커와 매수 가격을 정확히 입력해주세요.")
        else:
            # 데이터 로드 (넉넉하게)
            start_search = calc_date - datetime.timedelta(days=60)
            end_search = calc_date + datetime.timedelta(days=1)
            
            with st.spinner("데이터 분석 중..."):
                df_calc = get_data(calc_ticker, start_search, end_search)
            
            if df_calc is not None and not df_calc.empty:
                # ATR 계산
                high_low = df_calc['High'] - df_calc['Low']
                high_close = (df_calc['High'] - df_calc['Close'].shift()).abs()
                low_close = (df_calc['Low'] - df_calc['Close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                df_calc['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
                
                # 날짜 매칭
                target_date_str = calc_date.strftime("%Y-%m-%d")
                row = df_calc.loc[df_calc['Date'] == target_date_str]
                
                if row.empty:
                    row = df_calc.iloc[[-1]]
                    st.toast(f"⚠️ {target_date_str} 데이터가 없어 최근일({row['Date'].values[0]}) 기준으로 계산합니다.")

                atr_val = row['ATR'].values[0]
                
                if pd.isna(atr_val):
                    st.error("데이터 부족으로 ATR을 계산할 수 없습니다.")
                else:
                    # --- A. ATR 기준 계산 ---
                    atr_sl_price = calc_price - (atr_val * calc_atr_sl)
                    atr_tp_price = calc_price + (atr_val * calc_atr_tp)
                    
                    # 실제 변동폭 % 환산
                    atr_sl_pct = ((calc_price - atr_sl_price) / calc_price) * 100
                    atr_tp_pct = ((atr_tp_price - calc_price) / calc_price) * 100
                    
                    # --- B. 고정 % 기준 계산 ---
                    pct_sl_price = calc_price * (1 - calc_pct_sl / 100)
                    pct_tp_price = calc_price * (1 + calc_pct_tp / 100)
                    
                    # --- 결과 출력 ---
                    st.markdown(f"#### 📊 분석 결과 (진입가: **${calc_price:.2f}**)")
                    st.caption(f"📅 기준일 변동성(ATR): **${atr_val:.2f}**")

                    res_col1, res_col2 = st.columns(2)
                    
                    # [왼쪽] ATR 결과
                    with res_col1:
                        st.info(f"🌊 **ATR 기준 (SL x{calc_atr_sl} / TP x{calc_atr_tp})**")
                        st.metric("🚀 익절 목표가", f"${atr_tp_price:.2f}", f"+{atr_tp_pct:.2f}%")
                        st.metric("📉 손절 방어선", f"${atr_sl_price:.2f}", f"-{atr_sl_pct:.2f}%", delta_color="inverse")
                        
                        if atr_sl_pct > calc_pct_sl:
                            st.warning(f"⚠️ 변동성이 큽니다! (ATR 손절폭 -{atr_sl_pct:.1f}% > 고정 -{calc_pct_sl}%)")

                    # [오른쪽] 고정 % 결과
                    with res_col2:
                        st.success(f"🛑 **고정 비율 (SL -{calc_pct_sl}% / TP +{calc_pct_tp}%)**")
                        st.metric("🚀 익절 목표가", f"${pct_tp_price:.2f}", f"+{calc_pct_tp:.2f}%")
                        st.metric("📉 손절 방어선", f"${pct_sl_price:.2f}", f"-{calc_pct_sl:.2f}%", delta_color="inverse")
                        
            else:
                st.error("데이터를 불러올 수 없습니다.")

# --- 탭 6: 펀더멘털 (주가 vs EPS) ---
with tab6:
    st.markdown("### 📊 펀더멘털 & EPS 추세 분석")
    st.caption("주가(Price) 흐름과 기업의 **EPS(주당순이익)** 추이를 함께 비교합니다.")

    col_f1, col_f2 = st.columns([1, 3])
    
    with col_f1:
        default_ticker = st.session_state.get("signal_ticker", "NVDA")
        f_ticker = st.text_input("분석할 티커", value=default_ticker, key="fund_ticker")
        f_years = st.slider("조회 기간 (년)", 1, 5, 3, key="fund_years")
        
        korea_period = "분기(Quarter)"
        if f_ticker.endswith(".KS") or f_ticker.endswith(".KQ"):
            korea_period = st.radio("🇰🇷 실적 기준 선택", ["연간(Annual)", "분기(Quarter)"])
        
        st.info("""
        **차트 보는 법:**
        - **⚫ 회색선 (Left):** 주가 (Price)
        - **🔵 파란선 (Right):** EPS (주당순이익)
        
        ※ EPS를 찾지 못할 경우 '순이익'으로 대체되며 제목에 표시됩니다.
        """)

    with col_f2:
        if st.button("📉 데이터 가져오기", type="primary"):
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import yfinance as yf
            import requests
            import datetime

            # -----------------------------------------------------------
            # 🇰🇷 한국 주식 로직 (네이버 금융 + EPS Line Chart)
            # -----------------------------------------------------------
            if f_ticker.endswith(".KS") or f_ticker.endswith(".KQ"):
                st.subheader(f"🇰🇷 {f_ticker} 주가 vs EPS ({korea_period})")
                code = f_ticker.split('.')[0]
                url = f"https://finance.naver.com/item/main.naver?code={code}"
                
                try:
                    # 1. 재무 데이터 크롤링
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    dfs = pd.read_html(response.text, encoding='euc-kr')
                    
                    df_fin = None
                    for df in dfs:
                        # 매출액이나 영업이익이 있는 표 찾기
                        if df.shape[1] > 3 and df.iloc[:, 0].astype(str).str.contains("매출액|영업이익").any():
                            df_fin = df
                            break
                    
                    if df_fin is not None:
                        # 컬럼 중복 처리
                        raw_cols = [c[1] for c in df_fin.columns]
                        new_cols = []
                        counts = {}
                        for col in raw_cols:
                            if col in counts: counts[col] += 1; new_cols.append(f"{col}.{counts[col]}")
                            else: counts[col] = 0; new_cols.append(col)
                        df_fin.columns = new_cols
                        df_fin.set_index(df_fin.columns[0], inplace=True)

                        # 2. 데이터 분류 (연간 vs 분기)
                        target_cols = []
                        if "연간" in korea_period:
                            target_cols = [c for c in df_fin.columns[:4]] 
                        else:
                            target_cols = [c for c in df_fin.columns[4:]]

                        # [핵심 수정] EPS 우선 검색 로직
                        # 네이버 금융에서 EPS 표기법들을 순차적으로 찾습니다.
                        candidates = ["EPS(원)", "지배주주EPS(원)", "EPS"] 
                        row_name = None
                        is_eps = False
                        
                        for cand in candidates:
                            # 부분 일치 검색
                            matches = df_fin.index[df_fin.index.str.contains(cand, na=False)]
                            if len(matches) > 0:
                                row_name = matches[0] # 첫 번째 매칭된 행 이름 사용
                                is_eps = True
                                break
                        
                        # EPS가 정 없으면 당기순이익으로 대체 (그래프라도 보여주기 위함)
                        if row_name is None:
                            row_name = "당기순이익"
                            if df_fin.index.str.contains(row_name).any():
                                st.warning(f"⚠️ 'EPS' 데이터를 찾을 수 없어 '{row_name}'으로 대체합니다.")
                            else:
                                st.error("재무 데이터에서 실적 항목을 찾을 수 없습니다.")
                                st.stop()

                        # 데이터 추출
                        eps_row = df_fin.loc[row_name][target_cols]
                        
                        # 데이터 정제
                        dates = []
                        values = []
                        
                        for col, val in eps_row.items():
                            try:
                                clean_date_str = col.split('(')[0].strip().replace('(E)', '')
                                dt = datetime.datetime.strptime(clean_date_str, "%Y.%m")
                                dt = dt.replace(day=15)
                                
                                clean_val = float(str(val).replace(',', '').strip())
                                
                                dates.append(dt)
                                values.append(clean_val)
                            except: pass
                        
                        # 3. 차트 그리기
                        if dates:
                            start_d_price = min(dates) - datetime.timedelta(days=90)
                            end_d_price = datetime.date.today()
                            df_price = get_data(f_ticker, start_d_price, end_d_price)

                            fig, ax1 = plt.subplots(figsize=(10, 5))

                            # 축 1: 주가 (회색)
                            ax1.set_xlabel('Date')
                            ax1.set_ylabel('Price (KRW)', color='gray')
                            ax1.plot(df_price['Date'], df_price['Close'], color='gray', alpha=0.5, linewidth=1.5, label='Stock Price', zorder=1)
                            ax1.tick_params(axis='y', labelcolor='gray')

                            # 축 2: 실적 (EPS면 파란색, 순이익이면 빨간색)
                            ax2 = ax1.twinx()
                            
                            color = 'blue' if is_eps else 'crimson'
                            label_name = f"EPS (Won)" if is_eps else f"{row_name} (Net Income)"
                            
                            ax2.set_ylabel(label_name, color=color)
                            ax2.plot(dates, values, color=color, marker='o', linestyle='-', linewidth=2, markersize=6, label=label_name, zorder=2)
                            
                            for d, v in zip(dates, values):
                                ax2.text(d, v, f"{v:,.0f}", ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

                            ax2.tick_params(axis='y', labelcolor=color)
                            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                            
                            plt.title(f"{f_ticker} Price vs {label_name}", fontsize=15)
                            ax1.grid(True, alpha=0.3)
                            
                            lines1, labels1 = ax1.get_legend_handles_labels()
                            lines2, labels2 = ax2.get_legend_handles_labels()
                            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

                            st.pyplot(fig)
                            
                            st.write(f"#### 📋 상세 재무제표 ({row_name})")
                            st.dataframe(df_fin.loc[[row_name]][target_cols], use_container_width=True)
                            
                            if any("(E)" in c for c in target_cols):
                                st.caption("※ (E)는 컨센서스(예상치) 입니다.")
                                
                        else:
                            st.warning("유효한 날짜 데이터를 찾을 수 없습니다.")

                    else:
                        st.warning("재무제표 데이터를 찾을 수 없습니다.")

                except Exception as e:
                    st.error(f"분석 실패: {e}")

            # -----------------------------------------------------------
            # 🇺🇸 미국 주식 로직 (기존 유지)
            # -----------------------------------------------------------
            else:
                st.subheader(f"🇺🇸 {f_ticker} Earnings Surprise (Est vs Actual)")
                with st.spinner("미국 주식 데이터 분석 중..."):
                    try:
                        end_d = datetime.date.today()
                        start_d = end_d - datetime.timedelta(days=365 * f_years)
                        df_price = get_data(f_ticker, start_d, end_d)
                        
                        tick = yf.Ticker(f_ticker)
                        df_eps = tick.get_earnings_dates()
                        
                        if df_eps is not None and not df_eps.empty:
                            df_eps = df_eps.sort_index()
                            if df_eps.index.tz is not None: df_eps.index = df_eps.index.tz_localize(None)
                            df_eps = df_eps[df_eps.index >= pd.Timestamp(start_d)]
                            
                            if df_eps.empty:
                                st.warning("조회 기간 내 EPS 데이터가 없습니다.")
                            else:
                                fig, ax1 = plt.subplots(figsize=(10, 5))
                                ax1.set_xlabel('Date')
                                ax1.set_ylabel('Price ($)', color='black')
                                ax1.plot(df_price['Date'], df_price['Close'], color='black', alpha=0.2, label='Price')
                                
                                ax2 = ax1.twinx()
                                ax2.set_ylabel('EPS ($)', color='blue')
                                if 'EPS Estimate' in df_eps.columns:
                                    ax2.plot(df_eps.index, df_eps['EPS Estimate'], color='blue', marker='o', linestyle='--', alpha=0.6, label='Estimate')
                                if 'Reported EPS' in df_eps.columns:
                                    actual_data = df_eps.dropna(subset=['Reported EPS'])
                                    ax2.plot(actual_data.index, actual_data['Reported EPS'], color='green', marker='D', linestyle='-', markersize=8, label='Actual')

                                ax2.tick_params(axis='y', labelcolor='green')
                                plt.title(f"{f_ticker} Price vs Earnings Surprise")
                                ax1.grid(True, alpha=0.3)
                                lines1, labels1 = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                                st.pyplot(fig)
                                
                                if 'Reported EPS' in df_eps.columns:
                                    last_row = df_eps.dropna(subset=['Reported EPS']).iloc[-1]
                                    est, act = last_row['EPS Estimate'], last_row['Reported EPS']
                                    if pd.notna(est) and pd.notna(act):
                                        surprise = act - est
                                        st.markdown(f"#### 📢 최근 실적: 예상 ${est:.2f} vs 실제 ${act:.2f} ({'Beat' if surprise>0 else 'Miss'})")
                        else:
                            st.warning("EPS 추정치 데이터가 없습니다.")
                    except Exception as e:
                        st.error(f"오류 발생: {e}")

# --- 탭 7: 실험실 optuna AI ---
with tab7:
    st.subheader("🤖 AI 지능형 풀옵션 최적화")
    st.info("Optuna AI가 이평선, 추세, 익/손절, ATR 등 **모든 변수**를 동시에 조합하여 최적의 꿀통을 찾아냅니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.number_input("AI 탐색 횟수 (다중 변수는 최소 100회 이상 권장)", 50, 2000, 200)
    with col2:
        target_score = st.selectbox("최적화 목표 선택", [
            "수익률 (%)", 
            "다중 목적 (수익률⬆️ + MDD⬇️)", 
            "Profit Factor", 
            "승률 (%)"
        ])

    # 1️⃣ 첫 번째 버튼: AI 탐색 실행 후 '기억 장치'에 결과 저장
    if st.button("🚀 AI 최적화 시작"):
        with st.spinner("데이터 로딩 및 AI 지능형 탐색 진행 중..."):
            safe_cash = st.session_state.get('initial_cash', 5000000)
            safe_fee = st.session_state.get('fee_bps', 25)
            safe_slip = st.session_state.get('slip_bps', 1)
            safe_behavior = st.session_state.get('strategy_behavior', "1")
            safe_hold = st.session_state.get('min_hold_days', 0)

            ma_pool = [1] + list(range(5, 121, 5))
            
            base_full, x_sig_full, x_trd_full, ma_dict, _, _ = prepare_base(
                signal_ticker, trade_ticker, market_ticker, start_date, end_date, ma_pool
            )
            
            if base_full is None or base_full.empty:
                st.error("데이터를 불러오는 데 실패했습니다.")
            else:
                if target_score == "다중 목적 (수익률⬆️ + MDD⬇️)":
                    study = optuna.create_study(directions=["maximize", "minimize"])
                else:
                    study = optuna.create_study(direction="maximize")
                
                study.optimize(lambda trial: optuna_objective(
                    trial, base_full, x_sig_full, x_trd_full, ma_dict, 
                    safe_cash, safe_fee, safe_slip, safe_behavior, safe_hold, target_score
                ), n_trials=n_trials)
                
                # 💡 [핵심 해결책] 찾아낸 결과를 st.session_state에 저장!
                st.session_state["optuna_study"] = study
                st.session_state["optuna_target"] = target_score
                st.success("🎉 AI 최적화 완료! 아래 결과를 확인하세요.")

    # 2️⃣ 두 번째 블록: 저장된 결과가 있으면 화면에 띄우고 "적용 버튼" 생성 (첫 번째 버튼의 바깥)
    if "optuna_study" in st.session_state:
        st.divider()
        study = st.session_state["optuna_study"]
        t_score = st.session_state["optuna_target"]
        
        # 💡 [핵심 해결책] 에러를 방지하기 위한 "콜백(Callback)" 함수 정의
        # 화면을 그리기 전에 미리 값을 바꿔치기해서 Streamlit이 불평하지 못하게 만듭니다.
        def apply_optuna_callback(params_dict):
            for k, v in params_dict.items():
                st.session_state[k] = v
            st.session_state["preset_name_selector"] = "직접 설정"
            
        if t_score == "다중 목적 (수익률⬆️ + MDD⬇️)":
            st.write("### 🏆 AI가 찾아낸 최적의 타협점들 (Pareto Front)")
            st.info("수익률과 MDD는 반비례합니다. AI가 찾아낸 훌륭한 **'공격형 ~ 안정형'** 조합들 중 마음에 드는 것을 선택하세요!")
            
            best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
            
            for i, t in enumerate(best_trials):
                ret_val = t.values[0]
                mdd_val = t.values[1]
                st.write(f"#### 💎 [후보 {i+1}] 수익률: `{ret_val:.2f}%` / MDD: `-{mdd_val:.2f}%`")
                st.json(t.params)
                
                # 💡 [수정] on_click 속성을 사용하여 콜백 함수로 넘겨줍니다.
                st.button(
                    f"[후보 {i+1}] 이 설정 적용하기", 
                    key=f"apply_multi_{i}", 
                    on_click=apply_optuna_callback, 
                    args=(t.params,)
                )
                    
        else:
            st.write(f"### 🏆 최고 {t_score}: {study.best_value}")
            st.write("#### 💡 AI가 찾아낸 기적의 조합")
            st.json(study.best_params)
            
            # 💡 [수정] on_click 속성을 사용하여 콜백 함수로 넘겨줍니다.
            st.button(
                "이 설정 바로 적용하기", 
                key="apply_optuna_single", 
                on_click=apply_optuna_callback, 
                args=(study.best_params,)
            )

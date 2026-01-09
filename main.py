import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import google.generativeai as genai

# 모듈 불러오기 (파일 경로가 맞는지 확인 필요)
from modules.utils import load_saved_strategies, save_strategy_to_file, delete_strategy_from_file, parse_choices
from modules.data_loader import get_data, get_fundamental_info
from modules.strategy import prepare_base, check_signal_today, backtest_fast, summarize_signal_today, auto_search_train_test, apply_opt_params
from modules.llm_advisor import ask_gemini_analysis, ask_gemini_chat, ask_gemini_comprehensive_analysis

st.set_page_config(page_title="QuantLab: Modular Ver.", page_icon="⚡", layout="wide")

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
        "bb_entry_type": "상단선 돌파 (추세)", "bb_exit_type": "중심선(MA) 이탈"
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

_init_default_state()

# 기본 프리셋 데이터
PRESETS = {
    "SOXL 도전 전략": {"signal_ticker": "SOXL", "trade_ticker": "SOXL", "offset_cl_buy": 1, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 20, "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 20, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 5, "offset_compare_long": 20, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
    "SOXL 안전 전략": {"signal_ticker": "SOXL", "trade_ticker": "SOXL", "offset_cl_buy": 10, "buy_operator": "<", "offset_ma_buy": 10, "ma_buy": 60, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 20, "ma_compare_short": 10, "offset_compare_long": 50, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
    "SOXL 극도전 전략": {"signal_ticker": "SOXL", "trade_ticker": "SOXL", "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 5, "ma_buy": 5, "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 10, "ma_sell": 120, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 20, "offset_compare_long": 50, "ma_compare_long": 120, "stop_loss_pct": 49.0, "take_profit_pct": 25.0},
    "TSLL 안전 전략": {"signal_ticker": "TSLL", "trade_ticker": "TSLL", "offset_cl_buy": 20, "buy_operator": "<", "offset_ma_buy": 5, "ma_buy": 10, "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 60, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 20, "ma_compare_short": 50, "offset_compare_long": 20, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 20.0},
}

# 로컬 파일에 저장된 전략이 있다면 합치기
saved_strategies = load_saved_strategies()
if saved_strategies:
    PRESETS.update(saved_strategies)

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
# 2. 사이드바 (설정 & 저장) - [수정 완료 구간]
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
    
    # 구분선 (들여쓰기 수정됨)
    st.divider()

    # 전략 저장/삭제 메뉴 (들여쓰기 수정됨)
    with st.expander("💾 전략 저장/삭제"):
        save_name = st.text_input("새 전략 이름 입력")
        
        if st.button("현재 설정 저장하기"):
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
                    "use_rsi_filter", "rsi_period", "rsi_max"
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

    # 구분선 (들여쓰기 수정됨)
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
start_date = col4.date_input("시작일", value=datetime.date(2020, 1, 1))
end_date = col5.date_input("종료일", value=datetime.date.today())

with st.expander("📈 상세 설정 (Offset, 비용 등)", expanded=True):
    tabs = st.tabs(["📊 이평선 설정", "🚦 시장 필터", "🌊 볼린저 밴드", "🛡️ 리스크/기타"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📥 매수")
            ma_buy = st.number_input("매수 이평", key="ma_buy", step=1, min_value=1)
            offset_ma_buy = st.number_input("매수 이평 Offset", key="offset_ma_buy", step=1)
            offset_cl_buy = st.number_input("매수 종가 Offset", key="offset_cl_buy", step=1)
            buy_operator = st.selectbox("매수 부호", [">", "<"], key="buy_operator")
            use_trend_in_buy = st.checkbox("매수 추세 필터", key="use_trend_in_buy")
        with c2:
            st.markdown("#### 📤 매도")
            ma_sell = st.number_input("매도 이평", key="ma_sell", step=1, min_value=1)
            offset_ma_sell = st.number_input("매도 이평 Offset", key="offset_ma_sell", step=1)
            offset_cl_sell = st.number_input("매도 종가 Offset", key="offset_cl_sell", step=1)
            sell_operator = st.selectbox("매도 부호", ["<", ">"], key="sell_operator")
            use_trend_in_sell = st.checkbox("매도 역추세 필터", key="use_trend_in_sell")
        
        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 📈 추세선")
            ma_compare_short = st.number_input("추세 Short", key="ma_compare_short", step=1, min_value=1)
            offset_compare_short = st.number_input("추세 Short Offset", key="offset_compare_short", step=1)
        with c4:
            st.markdown("#### .")
            ma_compare_long = st.number_input("추세 Long", key="ma_compare_long", step=1, min_value=1)
            offset_compare_long = st.number_input("추세 Long Offset", key="offset_compare_long", step=1)

    with tabs[1]:
        st.markdown("#### 🚦 시장 필터 (Market Filter)")
        st.write("시장 지수(예: SPY)가 이평선 위에 있을 때만 매수합니다.")
        use_market_filter = st.checkbox("시장 필터 사용", key="use_market_filter")
        market_ma_period = st.number_input("시장 이평선 기간", value=200, step=10, key="market_ma_period")

    with tabs[2]:
        st.markdown("#### 🌊 볼린저 밴드 (Volatility Breakout)")
        st.write("이평선 매매 대신 볼린저 밴드 돌파 전략을 사용합니다.")
        use_bollinger = st.checkbox("볼린저 밴드 사용", key="use_bollinger")
        c_b1, c_b2 = st.columns(2)
        bb_period = c_b1.number_input("밴드 기간", value=20, key="bb_period")
        bb_std = c_b2.number_input("밴드 승수 (Std Dev)", value=2.0, step=0.1, key="bb_std")
        bb_entry_type = st.selectbox("매수 기준", ["상단선 돌파 (추세)", "하단선 이탈 (역추세)", "중심선 돌파"], key="bb_entry_type")
        bb_exit_type = st.selectbox("매도 기준", ["중심선(MA) 이탈", "상단선 복귀", "하단선 이탈"], key="bb_exit_type")
        if use_bollinger:
            st.info("ℹ️ 활성화 시 '이평선 매매' 조건은 무시됩니다.")

    with tabs[3]:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### 🛡️ 리스크")
            stop_loss_pct = st.number_input("손절 (%)", step=0.5, key="stop_loss_pct")
            take_profit_pct = st.number_input("익절 (%)", step=0.5, key="take_profit_pct")
            min_hold_days = st.number_input("최소 보유일", step=1, key="min_hold_days")
        with c6:
            st.markdown("#### ⚙️ 기타")
            strategy_behavior = st.selectbox("행동 패턴", ["1. 포지션 없으면 매수 / 보유 중이면 매도", "2. 매수 우선", "3. 관망"], key="strategy_behavior")
            fee_bps = st.number_input("수수료 (bps)", value=25, step=1, key="fee_bps")
            slip_bps = st.number_input("슬리피지 (bps)", value=5, step=1, key="slip_bps")
            seed = st.number_input("랜덤 시드", value=0, step=1)
            if seed > 0: random.seed(seed)
        
        st.divider()
        st.markdown("#### 🔮 보조지표 설정")
        c_r1, c_r2 = st.columns(2)
        rsi_p = c_r1.number_input("RSI 기간 (Period)", 14, step=1, key="rsi_period")
        u_rsi = st.checkbox("RSI 필터 적용 (매수시 과열 방지)", key="use_rsi_filter")
        if u_rsi:
            rsi_max = c_r2.number_input("RSI 과매수 기준", 70, key="rsi_max")

# ==========================================
# 4. 기능 탭
# ==========================================
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏢 기업 정보", "🎯 시그널", "📚 PRESETS", "🧪 백테스트", "🧬 실험실"])

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

with tab2:
    if st.button("📚 모든 프리셋 일괄 점검"):
        rows = []
        with st.spinner("모든 전략을 시뮬레이션 중입니다..."):
            for name, p in PRESETS.items():
                t = p.get("signal_ticker", p.get("trade_ticker"))
                res = summarize_signal_today(get_data(t, start_date, end_date), p)
                rows.append({
                    "전략": name, "티커": t, "시그널": res["label"], 
                    "최근 BUY": res["last_buy"], "최근 SELL": res["last_sell"], "최근 HOLD": res["last_hold"]
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

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
                                bb_entry_type=st.session_state.bb_entry_type, bb_exit_type=st.session_state.bb_exit_type)
            st.session_state["bt_result"] = res
            if "ai_analysis" in st.session_state: del st.session_state["ai_analysis"]
            st.rerun()
        else: st.error("데이터 로딩 실패")

    if "bt_result" in st.session_state:
        res = st.session_state["bt_result"]
        if res:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("총 수익률", f"{res['수익률 (%)']}%", delta_color="normal")
            k2.metric("MDD (최대낙폭)", f"{res['MDD (%)']}%", delta_color="inverse")
            k3.metric("승률", f"{res['승률 (%)']}%")
            k4.metric("Profit Factor", res['Profit Factor'])
            
            df_log = pd.DataFrame(res['매매 로그'])
            if not df_log.empty:
                initial_price = df_log['종가'].iloc[0]
                benchmark = (df_log['종가'] / initial_price) * 5000000
                drawdown = (df_log['자산'] - df_log['자산'].cummax()) / df_log['자산'].cummax() * 100

                chart_data = res.get("차트데이터", {})
                base_df = chart_data.get("base")
                
                # 차트 그리기
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25], 
                                    subplot_titles=("주가 & 매매타점 (Candle + MA)", "내 자산 vs 보유 전략 (Equity)", "MDD (%)"))

                if base_df is not None:
                    fig.add_trace(go.Candlestick(x=base_df['Date'], open=base_df['Open_sig'], high=base_df['High_sig'], low=base_df['Low_sig'], close=base_df['Close_sig'], name='가격(Signal)'), row=1, col=1)
                    
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
                st.markdown("### 🤖 제미니 퀀트 컨설턴트 (1:1 대화)")
                chat_container = st.container(height=300)
                for msg in st.session_state["chat_history"]:
                    with chat_container.chat_message(msg["role"]): st.write(msg["content"])

                if prompt := st.chat_input("전략에 대해 질문하세요!"):
                    st.session_state["chat_history"].append({"role": "user", "content": prompt})
                    with chat_container.chat_message("user"): st.write(prompt)
                    with chat_container.chat_message("assistant"):
                        current_p = f"매수:{ma_buy}MA, 매도:{ma_sell}MA, 손절:{stop_loss_pct}%"
                        response = ask_gemini_chat(prompt, res, current_p, trade_ticker, st.session_state["gemini_api_key"], st.session_state.get("selected_model_name"))
                        st.write(response)
                        st.session_state["chat_history"].append({"role": "assistant", "content": response})

                st.markdown("### 💾 결과 저장")
                csv = df_log.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 매매 로그 다운로드 (CSV)", data=csv, file_name=f'backtest_log_{trade_ticker}_{datetime.date.today()}.csv', mime='text/csv')

                st.divider()
                st.markdown("### 🤖 Gemini AI 전략 컨설팅")
                if st.button("✨ AI에게 분석 및 개선점 물어보기", type="primary"):
                    fd = get_fundamental_info(trade_ticker)
                    sl_txt = f"{stop_loss_pct}%" if stop_loss_pct > 0 else "미설정"
                    tp_txt = f"{take_profit_pct}%" if take_profit_pct > 0 else "미설정"
                    current_params = f"매수: {ma_buy}일 이평, 매도: {ma_sell}일 이평, 손절: {sl_txt}, 익절: {tp_txt}"
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
        cand_sell_op = st.text_input("매도 부호", "<,>")
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
        cand_stop = st.text_input("손절(%) 후보", "0, 5, 10, 20")
        cand_take = st.text_input("익절(%) 후보", "0, 10, 20")

    n_trials = st.number_input("시도 횟수", 10, 500, 50)
    split_ratio = st.slider("Train 비율", 0.5, 0.9, 0.7)
    
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
                    df_opt[col] = pd.to_numeric(df_opt[col], errors='ignore')
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

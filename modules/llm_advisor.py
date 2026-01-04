import google.generativeai as genai
import streamlit as st

def ask_gemini_analysis(summary, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key를 입력해주세요."
    try:
        genai.configure(api_key=api_key)
        m_name = model_name if model_name else "gemini-1.5-flash"
        model = genai.GenerativeModel(m_name)
        
        prompt = f"""
        당신은 상위 1% 퀀트 트레이더입니다. 
        이 전략은 '종가 매매(Market On Close)'를 기준으로 백테스트 되었습니다.

        [투자 대상]: {ticker}
        [전략 설정]: {params}
        
        [백테스트 결과]
        - 수익률: {summary.get('수익률 (%)')}%
        - MDD: {summary.get('MDD (%)')}%
        - 승률: {summary.get('승률 (%)')}%
        - Profit Factor: {summary.get('Profit Factor')}
        - 총 매매 횟수: {summary.get('총 매매 횟수')}회

        [요청사항]
        1. 📊 **성과 진단**: 이 전략의 장점과 치명적인 단점은 무엇인가요?
        2. 🛠️ **튜닝 가이드**: 지표(이평선, 볼린저 등)의 기간을 어떻게 조절하면 좋을까요?
        3. 💡 **종합 평가**: 실전 투자에 적합한가요? (추천/보류/비추천)
        """
        with st.spinner("🤖 Gemini가 전략을 분석 중입니다..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: return f"❌ Gemini 분석 오류: {e}"

def ask_gemini_chat(question, res, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key를 입력해주세요."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name if model_name else "gemini-1.5-flash")
        context = f"""
        당신은 월스트리트의 상위 1% 퀀트 전문가입니다. 다음 전략 데이터를 바탕으로 사용자의 질문에 답하세요.
        [데이터] 수익률: {res.get('수익률 (%)') or 0}%, MDD: {res.get('MDD (%)') or 0}%, 
        승률: {res.get('승률 (%)') or 0}%, PF: {res.get('Profit Factor') or 0}, 티커: {ticker}
        [설정] {params}
        사용자 질문: {question}
        냉철하고 논리적으로 트레이더의 관점에서 조언하세요.
        """
        response = model.generate_content(context)
        return response.text
    except Exception as e: return f"❌ 오류: {e}"

# [추가됨] 기업 분석용 함수
def ask_gemini_comprehensive_analysis(summary, fundamental, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key를 입력해주세요."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name if model_name else "gemini-1.5-flash")
        mkt_cap = f"{fundamental['MarketCap'] / 100000000:.2f}억" if fundamental['MarketCap'] else "N/A"
        
        prompt = f"""
        당신은 펀드매니저이자 퀀트 트레이더입니다. [기본적 분석]과 [기술적 백테스트]를 통합하여 조언하세요.

        1. 대상: {fundamental['Name']} ({ticker}) / {fundamental['Sector']} / 시총 {mkt_cap}
           - PER: {fundamental['PER']}, ROE: {fundamental['ROE']}
           - 개요: {fundamental['Description'][:300]}...
        2. 전략: {params}
        3. 성과: 수익 {summary.get('수익률 (%)')}%, MDD {summary.get('MDD (%)')}%

        [요청]
        1. 🏢 기업 건전성 (저평가/고평가 여부)
        2. 📈 전략 적합성 (변동성 고려)
        3. ⚖️ 최종 조언 (적극투자/관망/주의)
        """
        with st.spinner("🤖 Gemini가 통합 분석 중입니다..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: return f"❌ 오류: {e}"
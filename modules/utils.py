import streamlit as st
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Streamlit Secrets에서 설정값 가져오기
def get_google_sheet():
    try:
        # 1. 인증 정보 가져오기
        key_dict = json.loads(st.secrets["GCP_KEY"])
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        
        # 2. 시트 열기
        sheet_url = st.secrets["SHEET_URL"]
        sheet = client.open_by_url(sheet_url).sheet1
        return sheet
    except Exception as e:
        st.error(f"구글 시트 연결 실패: {e}")
        return None

def load_saved_strategies():
    sheet = get_google_sheet()
    if not sheet: return {}
    try:
        # 모든 기록 가져오기
        data = sheet.get_all_records()
        strategies = {}
        for row in data:
            # 엑셀의 각 줄을 딕셔너리로 변환
            name = row.get("StrategyName")
            if name:
                # JSON 문자열로 저장된 파라미터를 다시 딕셔너리로
                params = json.loads(row.get("Params"))
                strategies[name] = params
        return strategies
    except: return {}

def save_strategy_to_file(name, params):
    sheet = get_google_sheet()
    if not sheet: return
    
    try:
        # 기존에 같은 이름이 있으면 삭제하고 추가 (혹은 업데이트)
        # 편의상 그냥 아래에 추가하는 로직
        params_str = json.dumps(params, ensure_ascii=False)
        sheet.append_row([name, params_str, str(datetime.datetime.now())])
        st.toast(f"✅ 전략 '{name}' 구글 시트에 저장 완료!")
    except Exception as e:
        st.error(f"저장 실패: {e}")

def delete_strategy_from_file(name):
    # 삭제는 로직이 복잡해져서 (행을 찾아서 지워야 함)
    # 초보자 단계에서는 '구글 시트 가서 직접 지우세요'라고 안내하는 게 안전합니다.
    st.info("🗑️ 삭제는 구글 스프레드시트에서 직접 행을 지워주세요.")

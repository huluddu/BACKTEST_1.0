import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# -----------------------------------------------------------
# [설정] Streamlit Secrets의 변수명 (수정 금지)
SECRET_KEY_NAME = "GCP_KEY"     
SHEET_URL_NAME = "SHEET_URL"    
# -----------------------------------------------------------

def _get_sheet_connection():
    """Streamlit Secrets의 URL을 이용해 구글 시트에 바로 연결"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Secrets에 키가 있는지 확인
    if SECRET_KEY_NAME not in st.secrets:
        st.error(f"⚠️ 설정 오류: Streamlit Secrets에 '{SECRET_KEY_NAME}'가 없습니다.")
        return None
    if SHEET_URL_NAME not in st.secrets:
        st.error(f"⚠️ 설정 오류: Streamlit Secrets에 '{SHEET_URL_NAME}'가 없습니다.")
        return None

    try:
        # 인증 정보 가져오기
        secret_value = st.secrets[SECRET_KEY_NAME]
        
        # 문자열(JSON String)로 되어 있다면 파싱, 딕셔너리면 그대로 사용
        if isinstance(secret_value, str):
            key_dict = json.loads(secret_value)
        else:
            key_dict = dict(secret_value)
        
        # 줄바꿈 문자(\n) 처리 (필수)
        if "private_key" in key_dict:
            key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")

        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        
        # URL로 시트 열기
        target_url = st.secrets[SHEET_URL_NAME]
        sheet = client.open_by_url(target_url).sheet1
        return sheet
        
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 실패: {e}")
        return None

# -----------------------------------------------------------
# 저장/로드/삭제 함수 (기존 로직 유지)
# -----------------------------------------------------------

def load_saved_strategies():
    sheet = _get_sheet_connection()
    if sheet is None: return {}
    try:
        records = sheet.get_all_records()
        strategies = {}
        for row in records:
            if not row: continue
            name = row.get('Name')
            params_str = row.get('Params')
            if name and params_str:
                try: strategies[name] = json.loads(params_str)
                except: continue
        return strategies
    except: return {}

def save_strategy_to_file(name, params):
    sheet = _get_sheet_connection()
    if sheet is None: return

    try:
        # 헤더가 없으면 추가
        if not sheet.get_all_values():
            sheet.append_row(["Name", "Params"])

        # 저장 로직
        try:
            cell = sheet.find(name) # 이름 찾기
            # 있으면 업데이트
            params_str = json.dumps(params, ensure_ascii=False)
            sheet.update_cell(cell.row, 2, params_str)
        except gspread.exceptions.CellNotFound:
            # 없으면 추가
            params_str = json.dumps(params, ensure_ascii=False)
            sheet.append_row([name, params_str])

    except Exception as e:
        st.error(f"❌ 저장 실패: {e}")
        raise e

def delete_strategy_from_file(name):
    sheet = _get_sheet_connection()
    if sheet is None: return
    try:
        cell = sheet.find(name)
        sheet.delete_rows(cell.row)
        st.success(f"🗑️ 삭제 완료: {name}")
    except: 
        st.warning("삭제할 전략이 시트에 없습니다.")

def parse_choices(text_input, dtype="str"):
    if not text_input: return []
    parts = [p.strip() for p in text_input.split(',')]
    results = []
    for p in parts:
        try:
            if dtype == "int": results.append(int(p))
            elif dtype == "float": results.append(float(p))
            elif dtype == "bool": results.append(p.lower() == "true")
            else: results.append(p)
        except: continue
    return sorted(list(set(results)), key=lambda x: (isinstance(x, bool), x))

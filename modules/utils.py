import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# -----------------------------------------------------------
# [설정] secrets.toml에 있는 변수 이름들
SECRET_KEY_NAME = "GCP_KEY"     
SHEET_URL_NAME = "SHEET_URL"    
# -----------------------------------------------------------

def _get_sheet_connection():
    """Streamlit Secrets의 URL을 이용해 구글 시트에 바로 연결"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # 1. Secrets에 필요한 키가 있는지 확인
    if SECRET_KEY_NAME not in st.secrets:
        st.error(f"⚠️ secrets.toml에 '{SECRET_KEY_NAME}'가 없습니다.")
        return None
    if SHEET_URL_NAME not in st.secrets:
        st.error(f"⚠️ secrets.toml에 '{SHEET_URL_NAME}'가 없습니다.")
        return None

    try:
        # 2. 인증 정보 가져오기
        secret_value = st.secrets[SECRET_KEY_NAME]
        
        # 문자열이면 json 변환, 딕셔너리면 그대로 사용
        if isinstance(secret_value, str):
            key_dict = json.loads(secret_value)
        else:
            key_dict = dict(secret_value)
        
        # 줄바꿈 문자 처리 (\n)
        if "private_key" in key_dict:
            key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")

        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        
        # 3. URL로 바로 열기
        target_url = st.secrets[SHEET_URL_NAME]
        sheet = client.open_by_url(target_url).sheet1
        return sheet
        
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 실패: {e}")
        st.info("💡 secrets.toml의 SHEET_URL 주소가 정확한지 확인해주세요.")
        return None

# (이하 load, save, delete 함수들은 그대로 두시면 됩니다.)
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
        if not sheet.get_all_values(): sheet.append_row(["Name", "Params"])
        try:
            cell = sheet.find(name)
            params_str = json.dumps(params, ensure_ascii=False)
            sheet.update_cell(cell.row, 2, params_str)
        except gspread.exceptions.CellNotFound:
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
    except: st.warning("삭제할 전략이 시트에 없습니다.")

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

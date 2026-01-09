import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# -----------------------------------------------------------
# [설정] Streamlit Secrets 변수명
SECRET_KEY_NAME = "GCP_KEY"     
SHEET_URL_NAME = "SHEET_URL"    
# -----------------------------------------------------------

def _get_sheet_connection():
    """Streamlit Secrets의 URL을 이용해 구글 시트에 연결"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    if SECRET_KEY_NAME not in st.secrets:
        st.error(f"⚠️ 설정 오류: Secrets에 '{SECRET_KEY_NAME}'가 없습니다.")
        return None
    if SHEET_URL_NAME not in st.secrets:
        st.error(f"⚠️ 설정 오류: Secrets에 '{SHEET_URL_NAME}'가 없습니다.")
        return None

    try:
        secret_value = st.secrets[SECRET_KEY_NAME]
        
        if isinstance(secret_value, str):
            key_dict = json.loads(secret_value)
        else:
            key_dict = dict(secret_value)
        
        if "private_key" in key_dict:
            key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")

        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        
        target_url = st.secrets[SHEET_URL_NAME]
        sheet = client.open_by_url(target_url).sheet1
        return sheet
        
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 실패: {e}")
        return None

# -----------------------------------------------------------
# [핵심 수정] 에러 이름을 쓰지 않는 안전한 방식으로 변경
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

        # 1. 먼저 해당 이름의 셀을 찾아봅니다.
        cell = None
        try:
            cell = sheet.find(name)
        except Exception:
            # 못 찾으면 에러가 나는데, 이걸 무시하고 cell을 None으로 둡니다.
            cell = None

        params_str = json.dumps(params, ensure_ascii=False)

        # 2. 셀이 있으면(이미 저장된 전략) -> 업데이트
        if cell:
            # 해당 행의 2번째 열(Params)을 수정
            sheet.update_cell(cell.row, 2, params_str)
            # st.success(f"✅ 전략 업데이트 완료: {name}") # (메시지는 main.py에서 띄움)
            
        # 3. 셀이 없으면(새로운 전략) -> 추가
        else:
            sheet.append_row([name, params_str])
            # st.success(f"✅ 새 전략 저장 완료: {name}")

    except Exception as e:
        st.error(f"❌ 저장 실패: {e}")
        # 디버깅을 위해 에러 내용을 화면에 출력
        st.write(e)
        raise e

def delete_strategy_from_file(name):
    sheet = _get_sheet_connection()
    if sheet is None: return

    try:
        cell = None
        try:
            cell = sheet.find(name)
        except Exception:
            cell = None

        if cell:
            sheet.delete_rows(cell.row)
            st.success(f"🗑️ 삭제 완료: {name}")
        else:
            st.warning("삭제할 전략이 시트에 없습니다.")

    except Exception as e:
        st.error(f"삭제 오류: {e}")

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

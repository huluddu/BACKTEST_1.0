import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# -----------------------------------------------------------
# [설정] 구글 시트 이름
SHEET_NAME = "stock_strategies" 

# [중요] secrets.toml에 적은 헤더 이름 (대괄호 안에 적은 것)
# 예: [gcp_service_account] 라고 적으셨으면 아래와 같이 씁니다.
# 만약 [google_sheets] 라고 적으셨다면 st.secrets["google_sheets"]로 바꿔야 합니다.
SECRETS_KEY = "gcp_service_account" 
# -----------------------------------------------------------

def _get_sheet_connection():
    """Streamlit Secrets를 이용해 구글 시트에 연결"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # 1. Secrets에 해당 키가 있는지 확인
    if SECRETS_KEY not in st.secrets:
        st.error(f"⚠️ Streamlit Secrets에서 '{SECRETS_KEY}' 항목을 찾을 수 없습니다.")
        st.info("secrets.toml 파일의 대괄호[] 제목과 코드의 SECRETS_KEY가 일치하는지 확인해주세요.")
        return None

    try:
        # 2. 파일 경로가 아니라, Secrets에 있는 딕셔너리(JSON 내용)를 바로 사용
        # .from_json_keyfile_name() 대신 .from_json_keyfile_dict()를 사용해야 함
        key_dict = dict(st.secrets[SECRETS_KEY])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        
        client = gspread.authorize(creds)
        
        # 3. 시트 열기
        sheet = client.open(SHEET_NAME).sheet1
        return sheet
        
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 실패: {e}")
        st.info("💡 구글 시트 제목이 정확한지, 그리고 client_email 주소를 시트에 '편집자'로 초대했는지 확인해주세요.")
        return None

def load_saved_strategies():
    """구글 시트에서 전략 데이터를 불러옵니다."""
    sheet = _get_sheet_connection()
    if sheet is None:
        return {}

    try:
        records = sheet.get_all_records()
        strategies = {}
        
        for row in records:
            # 빈 행 스킵
            if not row: continue
            
            # 시트 헤더가 Name, Params라고 가정
            name = row.get('Name')
            params_str = row.get('Params')
            
            if name and params_str:
                try:
                    params = json.loads(params_str)
                    strategies[name] = params
                except:
                    continue
        return strategies

    except Exception as e:
        # 아직 데이터가 없거나 헤더 문제일 경우 빈 딕셔너리 반환
        return {}

def save_strategy_to_file(name, params):
    """구글 시트에 전략을 저장(추가/업데이트)합니다."""
    sheet = _get_sheet_connection()
    if sheet is None: return

    try:
        # 1. 헤더가 없는 경우 추가 (첫 실행 대비)
        if not sheet.get_all_values():
            sheet.append_row(["Name", "Params"])

        # 2. 이름 검색 후 업데이트 또는 추가
        try:
            cell = sheet.find(name)
            # 이미 있으면 업데이트 (2번째 열 = Params)
            params_str = json.dumps(params, ensure_ascii=False)
            sheet.update_cell(cell.row, 2, params_str)
            
        except gspread.exceptions.CellNotFound:
            # 없으면 새로 추가
            params_str = json.dumps(params, ensure_ascii=False)
            sheet.append_row([name, params_str])

    except Exception as e:
        st.error(f"❌ 저장 중 오류 발생: {e}")
        raise e

def delete_strategy_from_file(name):
    """구글 시트에서 전략을 삭제합니다."""
    sheet = _get_sheet_connection()
    if sheet is None: return

    try:
        cell = sheet.find(name)
        sheet.delete_rows(cell.row)
        st.success(f"🗑️ 구글 시트: '{name}' 삭제 완료!")
    except gspread.exceptions.CellNotFound:
        st.warning("삭제할 전략을 시트에서 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"삭제 중 오류: {e}")

def parse_choices(text_input, dtype="str"):
    """그리드 서치용 파싱 함수 (변경 없음)"""
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

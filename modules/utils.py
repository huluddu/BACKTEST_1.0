import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# -----------------------------------------------------------
# [설정] 구글 시트 제목 (URL에 있는 ID 대신 제목을 사용합니다)
# 구글 시트 파일명을 'stock_strategies'로 꼭 맞춰주세요!
SHEET_NAME = "stock_strategies" 

# [설정] secrets.toml의 대괄호 이름과 정확히 일치해야 합니다.
SECRETS_KEY = "gcp_service_account" 
# -----------------------------------------------------------

def _get_sheet_connection():
    """Streamlit Secrets를 이용해 구글 시트에 연결"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # 1. Secrets 확인
    if SECRETS_KEY not in st.secrets:
        st.error(f"⚠️ secrets.toml 파일에 [{SECRETS_KEY}] 섹션이 없습니다.")
        return None

    try:
        # 2. Secrets 내용을 딕셔너리로 가져오기
        key_dict = dict(st.secrets[SECRETS_KEY])
        
        # private_key의 줄바꿈 문자(\n) 처리 (Streamlit이 자동 처리하지만 안전장치)
        if "private_key" in key_dict:
            key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")

        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        
        # 3. 시트 열기
        sheet = client.open(SHEET_NAME).sheet1
        return sheet
        
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ 구글 시트를 찾을 수 없습니다: '{SHEET_NAME}'")
        st.info("1. 구글 시트 제목을 정확히 'stock_strategies'로 변경했는지 확인하세요.")
        st.info(f"2. '{key_dict.get('client_email')}' 이메일을 편집자로 초대했는지 확인하세요.")
        return None
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 에러: {e}")
        return None

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
                try:
                    strategies[name] = json.loads(params_str)
                except:
                    continue
        return strategies
    except Exception:
        return {}

def save_strategy_to_file(name, params):
    sheet = _get_sheet_connection()
    if sheet is None: return

    try:
        # 헤더 확인 및 생성 (Name, Params)
        if not sheet.get_all_values():
            sheet.append_row(["Name", "Params"])

        # 저장 로직
        try:
            cell = sheet.find(name)
            # 이미 있으면 업데이트
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
    except gspread.exceptions.CellNotFound:
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

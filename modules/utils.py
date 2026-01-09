import streamlit as st
import json
import gspread
import re
import datetime
from oauth2client.service_account import ServiceAccountCredentials

# 1. 구글 시트 연결 함수
def get_google_sheet():
    try:
        # Secrets에서 키 가져오기
        key_dict = json.loads(st.secrets["GCP_KEY"])
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
        client = gspread.authorize(creds)
        
        # 시트 주소로 열기
        sheet_url = st.secrets["SHEET_URL"]
        sheet = client.open_by_url(sheet_url).sheet1
        return sheet
    except Exception as e:
        # 연결 실패 시 에러는 로그로만 남기고 None 반환 (앱이 안 죽게)
        print(f"구글 시트 연결 오류: {e}")
        return None

# 2. 전략 불러오기 (구글 시트 -> 앱)
def load_saved_strategies():
    sheet = get_google_sheet()
    if not sheet: return {} # 연결 실패하면 빈 딕셔너리 반환
    
    try:
        data = sheet.get_all_records()
        strategies = {}
        for row in data:
            name = row.get("StrategyName")
            if name and row.get("Params"):
                try:
                    params = json.loads(str(row.get("Params")))
                    strategies[name] = params
                except: continue
        return strategies
    except: return {}

# 3. 전략 저장하기 (앱 -> 구글 시트)
def save_strategy_to_file(name, params):
    sheet = get_google_sheet()
    if not sheet: 
        st.error("구글 시트 연결 실패. Secrets 설정을 확인하세요.")
        return
    
    try:
        params_str = json.dumps(params, ensure_ascii=False)
        # 시간 기록도 같이
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([name, params_str, now])
        st.toast(f"✅ 구글 시트에 '{name}' 저장 완료!")
    except Exception as e:
        st.error(f"저장 실패: {e}")

# 4. 전략 삭제하기
def delete_strategy_from_file(name):
    st.info("🗑️ 구글 시트 연동 모드에서는 엑셀 파일에서 직접 행을 삭제해주세요.")
    return False

# 5. [중요] 기존 헬퍼 함수 (이게 없으면 에러 남!)
def parse_choices(text, cast="int"):
    if text is None: return []
    tokens = [t for t in re.split(r"[,\s]+", str(text).strip()) if t != ""]
    if not tokens: return []
    def _to_bool(s): return s.strip().lower() in ("1", "true", "t", "y", "yes")
    out = []
    for t in tokens:
        try:
            if cast == "int": out.append("same" if str(t).lower()=="same" else int(t))
            elif cast == "float": out.append(float(t))
            elif cast == "bool": out.append(_to_bool(t))
            else: out.append(str(t))
        except: continue
    seen = set()
    dedup = []
    for v in out:
        if (v if cast != "str" else (v,)) in seen: continue
        seen.add(v if cast != "str" else (v,))
        dedup.append(v)
    return dedup

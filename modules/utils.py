import json
import os
import re
import streamlit as st

STRATEGY_FILE = "my_strategies.json"

def load_saved_strategies():
    if not os.path.exists(STRATEGY_FILE): return {}
    try:
        with open(STRATEGY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_strategy_to_file(name, params):
    saved = load_saved_strategies()
    saved[name] = params
    with open(STRATEGY_FILE, "w", encoding="utf-8") as f: json.dump(saved, f, ensure_ascii=False, indent=4)
    st.toast(f"✅ 전략 '{name}' 저장 완료!")

def delete_strategy_from_file(name):
    saved = load_saved_strategies()
    if name in saved:
        del saved[name]
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f: json.dump(saved, f, ensure_ascii=False, indent=4)
        return True
    return False

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
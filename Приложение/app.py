# Импортируем модуль cli из web под именем stcli
from streamlit.web import cli as stcli
import sys
from streamlit import runtime

# Проверяем существование runtime
runtime.exists()

# ──────────────────────────────────────────────────────────────────────
# Само приложение
# ──────────────────────────────────────────────────────────────────────
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import pandas as pd
import streamlit as st

DB_PATH = Path("detections.db")

# ──────────────────────────────────────────────────────────────────────
# Хак/утилита: уникальные ключи для виджетов внутри каждой вкладки
# ──────────────────────────────────────────────────────────────────────
@contextmanager
def unique_widget_keys(prefix: str):
    import streamlit as _s
    to_patch = [
        "text_input", "text_area", "number_input", "date_input", "time_input",
        "color_picker", "selectbox", "multiselect", "radio", "toggle",
        "slider", "checkbox", "file_uploader", "camera_input", "button"
    ]
    originals = {}

    def _wrap(fn_name):
        fn = getattr(_s, fn_name)
        def _inner(label, *args, key=None, **kwargs):
            if key is None:
                lbl = str(label)
                key_gen = f"{prefix}:{fn_name}:{lbl}"
                return fn(label, *args, key=key_gen, **kwargs)
            return fn(label, *args, key=key, **kwargs)
        return _inner

    try:
        for name in to_patch:
            if hasattr(_s, name):
                originals[name] = getattr(_s, name)
                setattr(_s, name, _wrap(name))
        yield
    finally:
        for name, orig in originals.items():
            setattr(_s, name, orig)

# ──────────────────────────────────────────────────────────────────────
# Чтение БД и «Статистика»
# ──────────────────────────────────────────────────────────────────────
def _read_detections(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query("SELECT * FROM detections ORDER BY ts DESC", conn)
    except Exception:
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts", ascending=False)
    return df.reset_index(drop=True)

def _render_stats():
    st.header("📊 История всех детекций")
    df = _read_detections(DB_PATH)
    if df.empty:
        st.info("В базе данных нет детекций.")
        return
    preferred = [
        "ts", "kind", "source", "filename", "model",
        "plate_number", "region", "year_issued",
        "brand", "car_type", "color", "conf_avg",
        "label_en", "label_ru", "score",
        "price_min", "price_max", "currency"
    ]
    seen, show_cols = set(), []
    for c in preferred:
        if c in df.columns and c not in seen:
            show_cols.append(c); seen.add(c)
    if not show_cols:
        show_cols = list(dict.fromkeys(df.columns.tolist()))
    with st.expander("Фильтры"):
        if "kind" in df.columns:
            kinds = sorted(df["kind"].dropna().unique().tolist())
            selected = st.multiselect("Типы записей", kinds, default=kinds, key="stats:kinds")
            if selected:
                df = df[df["kind"].isin(selected)]
    st.dataframe(df[show_cols], use_container_width=True, height=520)
    st.download_button(
        "Скачать CSV",
        df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="detections_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="stats:csv",
    )

# ──────────────────────────────────────────────────────────────────────
# Ленивые обёртки для компонентов (чтобы импорт не падал при отсутствии cv2/tf)
# ──────────────────────────────────────────────────────────────────────
def _render_yolo_tab():
    try:
        from components.yolo_detection import render_yolo_detection
        render_yolo_detection(DB_PATH)
    except Exception as e:
        st.warning("Вкладка YOLO недоступна в этом окружении.")
        with st.expander("Техническая деталь"):
            st.code(repr(e))

def _render_plate_tab():
    try:
        from components.plate_assistant import render_plate_assistant
        render_plate_assistant(DB_PATH)
    except Exception as e:
        st.warning("Вкладка «Номер + ИИ-ассистент» недоступна в этом окружении.")
        with st.expander("Техническая деталь"):
            st.code(repr(e))

def _render_parts_tab():
    try:
        from components.parts_assistant import render_parts_assistant
        render_parts_assistant(DB_PATH)
    except Exception as e:
        st.warning("Вкладка «Запчасть + ИИ-ассистент» недоступна в этом окружении.")
        with st.expander("Техническая деталь"):
            st.code(repr(e))

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Auto Parts • Demo", layout="wide")
    tabs = st.tabs([
        "🔧 Детекция (YOLO)",
        "🪪 Номер + ИИ-ассистент",
        "🧩 Запчасть + ИИ-ассистент",
        "📊 Статистика",
    ])
    with tabs[0]:
        with unique_widget_keys("yolo"):
            _render_yolo_tab()
    with tabs[1]:
        with unique_widget_keys("plate"):
            _render_plate_tab()
    with tabs[2]:
        with unique_widget_keys("parts"):
            _render_parts_tab()
    with tabs[3]:
        _render_stats()

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

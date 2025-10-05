"""
components/license_plate_interface.py
"""
import cv2
import json
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import streamlit as st

from utils import npr
from utils.logger import logger


def _load(up) -> Optional[np.ndarray]:
    if up is None:
        return None
    return cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)


def _xml(**kw) -> str:
    root = ET.Element("CarAnalysis")
    for k, v in kw.items():
        el = ET.SubElement(root, k); el.text = v or ""
    return ET.tostring(root, encoding="utf-8").decode()


def render_license_plate_interface(db_session):
    st.header("🪪 Номер + ИИ-ассистент")

    up = st.file_uploader("Фото автомобиля", ["jpg", "jpeg", "png"])
    if st.button("Демо") and up is None:
        up = open("ex2.jpg", "rb")

    if up is None:
        st.info("Выберите файл.")
        return
    img = _load(up)
    if img is None:
        st.error("Неверное изображение."); return

    boxed, roi = npr.detect_plate(img)
    st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), width=640)

    plate, confs = npr.recognize_plate(roi)
    if not plate or "?" in plate:
        plate = npr.recognize_plate_gpt(img); confs = []

    with st.spinner("ИИ-ассистент анализирует…"):
        brand = npr.detect_brand(img)
        color = npr.detect_color(img)
        ctype = npr.detect_car_type(img)
        region, year = npr.analyze_russian_number(plate) if plate else ("", "")
        desc = npr.describe_car(img)

    st.subheader("Результат")
    st.markdown(f"""
* **Номер:** `{plate}`
* **Марка:** {brand}
* **Тип:** {ctype}
* **Цвет:** {color}
* **Регион:** {region}
* **Год выдачи:** {year}
""")
    st.markdown("**Описание:** " + desc)

    json_data = json.dumps({
        "plate_number": plate, "brand": brand, "car_type": ctype,
        "color": color, "region": region, "year_issued": year,
        "description": desc,
    }, ensure_ascii=False, indent=2)
    st.download_button("⬇️ JSON", json_data, "car_analysis.json", "application/json")
    st.download_button("⬇️ XML",
                       _xml(PlateNumber=plate, Brand=brand, CarType=ctype,
                            Color=color, Region=region, YearIssued=year,
                            Description=desc),
                       "car_analysis.xml", "application/xml")

    try:
        from utils.database import log_detections_bulk
        log_detections_bulk([{
            "class": "car_full_analysis",
            "confidence": 1.0,
            "extra": {
                "plate": plate, "brand": brand, "color": color,
                "car_type": ctype, "region": region, "year": year,
            },
        }], db_session)
    except Exception as e:
        logger.debug("DB log: %s", e)

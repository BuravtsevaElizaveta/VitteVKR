"""
components/main_interface.py
Стримлит-вкладка YOLO с выводом запчастей и безопасной обработкой форматов.
"""

import logging
import tempfile
import json
from typing import Dict, List, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from utils.image_processing import draw_boxes
from utils.video_processing import (
    process_uploaded_video,
    process_video_realtime,
    process_rtsp_stream,
)
from utils.database import log_detections_bulk
from utils.helpers import format_detections
from utils.chat_comment import summarize_detections
from utils.parts_info import get_part_info

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Загрузка YOLO…")
def load_model(opt: str, weights: str, device_raw: str):
    device = (device_raw or "cpu").strip().lower()
    if device not in ("cpu", "cuda", "mps"):
        device = "cpu"
    try:
        if opt == "YOLOv9":
            from models.yolo_v9 import YOLOv9
            return YOLOv9(weights, device)
        if opt == "YOLOv8":
            from models.yolo_v8 import YOLOv8
            return YOLOv8(weights, device)
    except ImportError:
        pass
    from ultralytics import YOLO
    model = YOLO(weights)
    if device in ("cuda", "mps"):
        try:
            model.to(device)
        except Exception:
            pass
    return model


class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, conf, thick, db):
        self.model, self.conf, self.thick, self.db = model, conf, thick, db
        self.counts: Dict[str, int] = {}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        dets = self.model.predict(img, conf=self.conf)
        img = draw_boxes(img, dets, self.conf, self.model, self.thick)
        for d in dets:
            cls = int(d[5])
            nm = self.model.model.names[cls]
            self.counts[nm] = self.counts.get(nm, 0) + 1

        # авто-комментарий каждые 60 детекций
        if sum(self.counts.values()) % 60 == 0 and self.counts:
            comment = summarize_detections(self.counts)
            st.session_state.setdefault("_live_comment", st.empty()).markdown(
                f"**💬 ИИ-ассистент:** {comment}"
            )
            self.counts.clear()

        # лог в БД раз в 30 кадров
        if sum(self.counts.values()) % 30 == 0:
            try:
                log_detections_bulk(dets, self.db)
            except Exception:
                pass
        return img


# ───────────────────────── «Изображение» ───────────────────────────────
def _render_image(model, conf, thick, db):
    img_file = st.file_uploader("Изображение (JPG/PNG)", ["jpg", "jpeg", "png"])
    if img_file is None:
        return

    pil = Image.open(img_file)
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    dets = model.predict(img, conf=conf)
    out = draw_boxes(img.copy(), dets, conf, model, thick)
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width=640)

    try:
        log_detections_bulk(dets, db)
    except Exception:
        pass

    # форматированные детекции могут быть строкой или списком словарей
    formatted_json = format_detections(dets, model)
    st.json(formatted_json)

    if isinstance(formatted_json, str):
        try:
            formatted_list: List[Any] = json.loads(formatted_json)
        except Exception:
            formatted_list = []
    else:
        formatted_list = formatted_json

    rows = []
    for item in formatted_list:
        if not isinstance(item, dict):
            continue
        cls_name = item.get("class") or item.get("label")
        if cls_name is None:
            continue
        info = get_part_info(cls_name)
        rows.append({
            "Запчасть": info["name"],
            "Уверенность": f"{item.get('confidence', 0):.2f}",
            "Описание": info["description"],
        })

    if rows:
        st.subheader("Информация по автозапчастям")
        st.table(rows)


# ───────────────────────── «Видео off-line» ────────────────────────────
def _render_video_offline(model, conf, thick, db):
    vid = st.file_uploader("Видео-файл (offline)", ["mp4", "mov", "avi"])
    if vid is None:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(vid.read())
        path = tmp.name
    st.info("⏳ Обработка видео…")
    out_path = process_uploaded_video(path, model, conf, thick, db)
    st.video(out_path, format="video/mp4", start_time=0)


# ───────────────────────── «Видео on-line» ─────────────────────────────
def _render_video_online(model, conf, thick, db):
    vid = st.file_uploader("Видео-файл (online)", ["mp4", "mov", "avi"])
    if vid is None:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(vid.read()); path = tmp.name
    st_frame = st.empty()
    for frame, _ in process_video_realtime(path, model, conf, thick, db):
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=640)


# ───────────────────────── «RTSP / WebRTC» ─────────────────────────────
def _render_rtsp(model, conf, thick, db):
    url = st.text_input("RTSP-URL (пусто → WebRTC)")
    if url:
        process_rtsp_stream(url, model, conf, thick, db)
    else:
        webrtc_streamer(
            key="yolo-webrtc",
            video_transformer_factory=lambda: VideoTransformer(model, conf, thick, db),
            rtc_configuration={"iceServers": [{"urls": []}]},
        )


# ───────────────────────── интерфейс вкладки ───────────────────────────
def render_detection_interface(opt, weights, device, conf, thick, db):
    if not weights:
        st.warning("Укажите путь к весам.")
        return
    model = load_model(opt, weights, device)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🖼 Изображение", "🎞 Видео offline", "⚡ Видео online", "📡 Live / RTSP"]
    )
    with tab1:
        _render_image(model, conf, thick, db)
    with tab2:
        _render_video_offline(model, conf, thick, db)
    with tab3:
        _render_video_online(model, conf, thick, db)
    with tab4:
        _render_rtsp(model, conf, thick, db)

# components/yolo_detection.py
"""
Вкладка «Детекция (YOLO)» — изображение и офлайн-видео.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time, json, uuid, tempfile
import numpy as np
import streamlit as st

# ── безопасный импорт cv2 ────────────────────────────────────────────
CV2_OK, CV2_ERR = True, None
try:
    import cv2  # должен прийти из opencv-python-headless
except Exception as e:
    CV2_OK, CV2_ERR = False, e

# ── ultralytics только если cv2 поднялся ────────────────────────────
UL_OK, UL_ERR = True, None
try:
    if CV2_OK:
        from ultralytics import YOLO
    else:
        UL_OK, UL_ERR = False, RuntimeError("Ultralytics пропущен: OpenCV не загрузился")
except Exception as e:
    UL_OK, UL_ERR = False, e

from utils.db import insert_detection


@st.cache_resource(show_spinner="Загрузка YOLO-весов…")
def _load_yolo(weights_path: str):
    if not UL_OK:
        raise RuntimeError(
            "Пакет ultralytics недоступен. Убедитесь, что в requirements.txt НЕТ opencv-python, "
            "а есть opencv-python-headless."
        )
    return YOLO(weights_path)


def _annotate(img_bgr: np.ndarray, result, names: Dict[int, str],
              conf_thr: float, thickness: int) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    out = img_bgr.copy()
    detected: List[Tuple[str, float]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out, detected
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)
    for (x1,y1,x2,y2), c, cl in zip(xyxy, conf, cls):
        if float(c) < conf_thr:
            continue
        label = names.get(int(cl), str(cl))
        detected.append((label, float(c)))
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), thickness)
        txt = f"{label} {float(c):.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0,255,0), -1)
        cv2.putText(out, txt, (x1+1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out, detected


def _save_summary_to_db(db_path: Path, filename: str, source: str,
                        model_name: str, dets: List[Tuple[str, float]]):
    if not dets:
        return
    labels: Dict[str, List[float]] = {}
    for lbl, cf in dets:
        labels.setdefault(lbl, []).append(cf)
    avg = sorted([(lbl, float(np.mean(cfs))) for lbl, cfs in labels.items()],
                 key=lambda x: x[1], reverse=True)
    top5 = avg[:5]
    insert_detection(
        db_path=db_path,
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        kind="yolo",
        source=source,
        filename=filename,
        model=model_name,
        label_en=top5[0][0] if top5 else None,
        label_ru=top5[0][0] if top5 else None,
        score=float(top5[0][1]) if top5 else None,
        json_top5=json.dumps([{"label": l, "score": s} for l, s in top5], ensure_ascii=False),
        price_min=None, price_max=None, currency=None,
        image_path=None,
        plate_number=None, region=None, year_issued=None,
        brand=None, car_type=None, color=None, conf_avg=None,
        extra_json=None
    )


def render_yolo_detection(db_path: Path):
    st.header("🔧 Детекция (YOLO)")

    # Если OpenCV не загрузился — объясняем причину и выходим
    if not CV2_OK:
        st.warning("OpenCV (cv2) недоступен. Детекция отключена.")
        with st.expander("Показать техническую деталь ошибки"):
            st.code(repr(CV2_ERR))
        st.info("Обычно это происходит, когда установлен пакет opencv-python вместо opencv-python-headless.")
        return
    if not UL_OK:
        st.warning("Ultralytics YOLO недоступен.")
        with st.expander("Показать техническую деталь ошибки"):
            st.code(repr(UL_ERR))
        return

    with st.sidebar:
        st.subheader("Выбор модели YOLO")
        model_choice = st.selectbox("Выберите модель YOLO", ["YOLOv8"], index=0, key="yolo:ver")
        weights_path = st.text_input("Путь к модели YOLOv8", "Приложение/models/YOLOv8.pt", key="yolo:weights")

        st.subheader("Опции процессора")
        device = st.radio("Устройство", ["CPU", "GPU"], horizontal=True, key="yolo:device")
        device = "cuda" if device == "GPU" else "cpu"

        st.subheader("Настройки детекции")
        conf_thr = st.slider("Порог уверенности", 0.0, 1.0, 0.25, 0.01, key="yolo:conf")
        thickness = st.slider("Толщина линии", 1, 10, 2, 1, key="yolo:th")

    tabs = st.tabs(["🖼️ Изображение", "🎞️ Видео (офлайн)"])

    # ── Изображение
    with tabs[0]:
        img_file = st.file_uploader("Изображение (JPG/PNG)", type=["jpg", "jpeg", "png"], key="yolo:img_upl")
        if img_file is not None:
            np_img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB),
                     caption="Загруженное изображение (уменьшено)", width=520)

            if st.button("Анализировать изображение", type="primary", key="yolo:img_run"):
                try:
                    model = _load_yolo(weights_path)
                except Exception as e:
                    st.error(str(e)); return

                res = model.predict(np_img, verbose=False, device=device, conf=conf_thr, iou=0.5)
                all_dets: List[Tuple[str, float]] = []
                out = np_img.copy()
                for r in res:
                    names = getattr(r, "names", {}) or {}
                    out, dets = _annotate(out, r, names, conf_thr, thickness)
                    all_dets.extend(dets)

                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Результат детекции", width=520)
                _save_summary_to_db(db_path, img_file.name, "image", model_choice, all_dets)
                st.success("Готово. Результат сохранён в БД.")

    # ── Видео (офлайн)
    with tabs[1]:
        vid_file = st.file_uploader("Видео (MP4/AVI/MOV/MKV)", type=["mp4", "avi", "mov", "mkv"], key="yolo:vid_upl")
        col1, col2 = st.columns(2)
        with col1:
            step = st.slider("Обрабатывать каждый N-й кадр", 1, 10, 2, 1, key="yolo:vid_step")
        with col2:
            max_frames = st.number_input("Макс. кадров (0 — все)", 0, 50000, 0, 1, key="yolo:vid_max")

        start = st.button("Запустить обработку видео", type="primary", key="yolo:vid_run")

        if vid_file is not None:
            st.video(vid_file, format="video/mp4")  # просто плеер

        if start and vid_file is not None:
            tmp = Path(tempfile.gettempdir()) / f"_uploaded_{uuid.uuid4().hex}.mp4"
            tmp.write_bytes(vid_file.getvalue())
            try:
                model = _load_yolo(weights_path)
            except Exception as e:
                st.error(str(e))
                try:
                    if tmp.exists(): tmp.unlink()
                except Exception:
                    pass
                return

            cap = cv2.VideoCapture(str(tmp))
            if not cap.isOpened():
                st.error("Не удалось открыть видео.")
                try:
                    if tmp.exists(): tmp.unlink()
                except Exception:
                    pass
                return

            frame_box = st.empty()
            progress = st.progress(0)
            info = st.empty()
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            processed = 0
            dets_all: List[Tuple[str, float]] = []

            i = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                i += 1
                if (i - 1) % max(step, 1) != 0:
                    continue

                res = model.predict(frame, verbose=False, device=device, conf=conf_thr, iou=0.5)
                out = frame.copy()
                for r in res:
                    names = getattr(r, "names", {}) or {}
                    out, dets = _annotate(out, r, names, conf_thr, thickness)
                    dets_all.extend(dets)

                frame_box.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Аннотированный поток", width=640)

                processed += 1
                if total > 0:
                    progress.progress(min(100, int(i / total * 100)))
                info.caption(f"Кадр {i}/{total or '…'} (обработано {processed})")

                if max_frames and processed >= max_frames:
                    break

            cap.release()
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass

            progress.empty(); info.empty()
            _save_summary_to_db(db_path, vid_file.name, "video", model_choice, dets_all)
            st.success("Обработка видео завершена и сохранена в БД.")


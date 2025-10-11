# components/yolo_detection.py
"""
Вкладка «Детекция (YOLO)»
— Изображение: детекция + аннотация.
— Видео (офлайн): загрузка локального видео, аннотированный «стрим» во вкладке.
— Запись в detections.db.

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time, json, uuid, tempfile
import numpy as np
import streamlit as st
import cv2

from utils.db import insert_detection

CV2_OK = True
CV2_ERR = None
ULTRA_OK = True
ULTRA_ERR = None

try:
    import cv2  # OpenCV
except Exception as e:
    CV2_OK = False
    CV2_ERR = e

try:
    # импортируйте ultralytics/torch только если cv2 есть
    if CV2_OK:
        from ultralytics import YOLO
    else:
        ULTRA_OK = False
        ULTRA_ERR = RuntimeError("Ultralytics пропущен, т.к. OpenCV не загрузился")
except Exception as e:
    ULTRA_OK = False
    ULTRA_ERR = e

@st.cache_resource(show_spinner="Загрузка YOLO-весов…")
def _load_yolo(weights_path: str):
    if not UL_OK:
        raise RuntimeError(
            "Пакет ultralytics не установлен. Установите: pip install ultralytics"
        )
    return YOLO(weights_path)


def _annotate(img_bgr: np.ndarray, result, names: Dict[int, str], conf_thr: float, thickness: int) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Рисуем боксы и собираем (label, conf) для БД."""
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


def _save_summary_to_db(db_path: Path, filename: str, source: str, model_name: str, dets: List[Tuple[str, float]]):
    """Агрегированная запись одной сессии детекций в detections.db."""
    if not dets:
        return
    labels = {}
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


def render_yolo_detection():
    st.header("🔧 Детекция (YOLO)")
    if not CV2_OK:
        st.warning(
            "Модуль OpenCV (cv2) недоступен в этом деплое. "
            "Детекция временно отключена, остальные вкладки работают."
        )
        with st.expander("Показать техническую деталь ошибки"):
            st.code(repr(CV2_ERR))
        return
    if not ULTRA_OK:
        st.warning("YOLO недоступен. Детекция отключена.")
        with st.expander("Показать техническую деталь ошибки"):
            st.code(repr(ULTRA_ERR))
        return

    tabs = st.tabs(["🖼️ Изображение", "🎞️ Видео (офлайн)"])

    # ─── Изображение ───
    with tabs[0]:
        img_file = st.file_uploader("Изображение (JPG/PNG)", type=["jpg", "jpeg", "png"], key="yolo:img_upl")
        if img_file is not None:
            img_bytes = img_file.read()
            np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            w_view = 520
            st.image(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB),
                     caption="Загруженное изображение (уменьшено)",
                     width=w_view)

            run = st.button("Анализировать изображение", type="primary", key="yolo:img_run")
            if run:
                try:
                    model = _load_yolo(weights_path)
                except Exception as e:
                    st.error(str(e))
                    return

                res = model.predict(np_img, verbose=False, device=device, conf=conf_thr, iou=0.5)
                all_dets: List[Tuple[str, float]] = []
                out = np_img.copy()
                for r in res:
                    names = getattr(r, "names", {}) or {}
                    out, dets = _annotate(out, r, names, conf_thr, thickness)
                    all_dets.extend(dets)

                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                         caption="Результат детекции",
                         width=w_view)

                _save_summary_to_db(db_path, img_file.name, "image", model_choice, all_dets)
                st.success("Готово. Результат сохранён в БД.")

    # ─── Видео (офлайн) ───
    with tabs[1]:
        vid_file = st.file_uploader("Видео (MP4/AVI/MOV)", type=["mp4", "avi", "mov", "mkv"], key="yolo:vid_upl")
        col1, col2 = st.columns(2)
        with col1:
            step = st.slider("Обрабатывать каждый N-й кадр", 1, 10, 2, 1, key="yolo:vid_step")
        with col2:
            max_frames = st.number_input("Макс. кадров (0 — все)", 0, 50000, 0, 1, key="yolo:vid_max")

        start = st.button("Запустить обработку видео", type="primary", key="yolo:vid_run")

        if vid_file is not None:
            st.video(vid_file, format="video/mp4")  # оригинал — просто плеер

        if start and vid_file is not None:
            # Без st.secrets: создаём временный путь в системном tmp
            tmp_dir = Path(tempfile.gettempdir())
            tmp_path = tmp_dir / f"_uploaded_{uuid.uuid4().hex}.mp4"
            tmp_path.write_bytes(vid_file.getvalue())

            try:
                model = _load_yolo(weights_path)
            except Exception as e:
                st.error(str(e))
                # Чистим временный файл
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return

            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                st.error("Не удалось открыть видео.")
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
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
                if not ret:
                    break
                i += 1
                if (i - 1) % max(step, 1) != 0:
                    continue

                res = model.predict(frame, verbose=False, device=device, conf=conf_thr, iou=0.5)
                out = frame.copy()
                for r in res:
                    names = getattr(r, "names", {}) or {}
                    out, dets = _annotate(out, r, names, conf_thr, thickness)
                    dets_all.extend(dets)

                frame_box.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                                caption="Аннотированный поток",
                                width=640)

                processed += 1
                if total > 0:
                    progress.progress(min(100, int(i / total * 100)))
                info.caption(f"Кадр {i}/{total or '…'} (обработано {processed})")

                if max_frames and processed >= max_frames:
                    break

            cap.release()
            # Чистим временный файл (для Py3.9 без missing_ok)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

            progress.empty()
            info.empty()

            _save_summary_to_db(db_path, vid_file.name, "video", model_choice, dets_all)
            st.success("Обработка видео завершена и сохранена в БД.")


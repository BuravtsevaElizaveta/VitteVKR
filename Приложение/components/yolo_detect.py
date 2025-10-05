# components/yolo_detect.py
# Детекция объектов через Ultralytics YOLO:
#  - Изображение (JPG/PNG)
#  - Локальное видео с «псевдостримом» (покадровый показ во вкладке)

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import io, time, tempfile, datetime as dt

import numpy as np
from PIL import Image
import streamlit as st

from utils.db import insert_detection

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    _YOLO_OK = False


def _draw_boxes(image: Image.Image, boxes_xyxy: np.ndarray, labels: List[str], confs: np.ndarray, thickness: int = 2) -> Image.Image:
    import cv2
    img = np.array(image.convert("RGB"))
    for (x1, y1, x2, y2), lab, c in zip(boxes_xyxy.astype(int), labels, confs):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 80, 20), thickness)
        text = f"{lab} {c:.2f}"
        cv2.putText(img, text, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 20), 1, cv2.LINE_AA)
    return Image.fromarray(img)


def _yolo_infer_frame(model: YOLO, frame_bgr: np.ndarray, conf: float, iou: float, thickness: int = 2) -> Tuple[Image.Image, Dict[str, Any]]:
    import cv2
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    res = model.predict(source=img, conf=conf, iou=iou, verbose=False)
    if not res:
        return pil, {"n": 0, "top": None}
    r = res[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return pil, {"n": 0, "top": None}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)
    names = [r.names[int(c)] for c in clss] if hasattr(r, "names") else [str(c) for c in clss]
    ann = _draw_boxes(pil, xyxy, names, confs, thickness=thickness)
    idx = int(np.argmax(confs))
    top = dict(label_en=names[idx], score=float(confs[idx]))
    return ann, {"n": len(xyxy), "top": top}


def _run_yolo_on_image(model: YOLO, img_bytes: bytes, conf: float, iou: float, thickness: int) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    res = model.predict(source=np.array(img), conf=conf, iou=iou, verbose=False)
    if not res:
        return {"annotated": img, "n": 0, "top": None}
    r = res[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return {"annotated": img, "n": 0, "top": None}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    names = [r.names[int(c)] for c in clss] if hasattr(r, "names") else [str(c) for c in clss]
    annotated = _draw_boxes(img, xyxy, names, confs, thickness=thickness)
    idx = int(np.argmax(confs))
    top = dict(label_en=names[idx], score=float(confs[idx]))
    return {"annotated": annotated, "n": len(xyxy), "top": top}


def _play_local_video_with_yolo(db_path: Path, model: YOLO, video_bytes: bytes, filename: str,
                                conf: float, iou: float, device_arg, display_width: int,
                                infer_every_n_frames: int, save_every_sec: float, write_annotated: bool,
                                thickness: int = 2):
    import cv2, os
    model.to(device_arg)
    tdir = tempfile.TemporaryDirectory()
    src_path = Path(tdir.name) / filename
    with open(src_path, "wb") as f: f.write(video_bytes)
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        st.error("Не удалось открыть видео."); tdir.cleanup(); return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    out_path = None; writer = None
    if write_annotated:
        out_path = Path(tdir.name) / f"annotated_{Path(filename).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (width, height))
    img_slot = st.empty(); info_slot = st.empty(); prog_bar = st.progress(0)
    last_save_t = 0.0; frame_idx = 0; start_t = time.time()
    st.session_state.setdefault("video_running", True)
    while st.session_state.get("video_running", False):
        ok, frame_bgr = cap.read()
        if not ok: break
        do_infer = (frame_idx % max(1, infer_every_n_frames) == 0)
        if do_infer:
            ann_pil, meta = _yolo_infer_frame(model, frame_bgr, conf=conf, iou=iou, thickness=thickness)
        else:
            ann_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)); meta = {"n": None, "top": None}
        img_slot.image(ann_pil, caption=f"{filename}", width=display_width)
        if writer is not None:
            ann_bgr = cv2.cvtColor(np.array(ann_pil), cv2.COLOR_RGB2BGR); writer.write(ann_bgr)
        prog_bar.progress(min(100, int(100.0*(frame_idx+1)/max(1,total_frames))))
        info = f"Кадр {frame_idx+1}/{total_frames} • FPS источника: {fps_src:.1f}"
        if meta["n"] is not None: info += f" • Детекций: {meta['n']}"
        info_slot.caption(info)
        now = time.time()
        if meta.get("top") is not None and (now - last_save_t) >= float(save_every_sec):
            last_save_t = now
            insert_detection(
                db_path=db_path, ts=dt.datetime.now().isoformat(timespec="seconds"),
                kind="yolo", source="video", filename=filename, model=str(getattr(model, "model", "")) or "YOLO",
                label_en=meta["top"]["label_en"], label_ru=meta["top"]["label_en"], score=float(meta["top"]["score"]),
                json_top5=None, price_min=None, price_max=None, currency=None, image_path=None,
                plate_number=None, region=None, year_issued=None, brand=None, car_type=None, color=None, conf_avg=None, extra_json=None
            )
        # синхронизация
        elapsed = time.time() - start_t; expected = (frame_idx + 1) / max(1e-6, fps_src)
        delay = expected - elapsed
        if delay > 0: time.sleep(min(0.2, delay))
        frame_idx += 1
    cap.release(); 
    if writer is not None: writer.release()
    info_slot.info("Воспроизведение завершено."); prog_bar.empty()
    if out_path and out_path.exists():
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Скачать аннотированное видео (MP4)", f.read(), file_name=out_path.name, mime="video/mp4")
    try: tdir.cleanup()
    except Exception: pass


def render_yolo_page(db_path: Path):
    st.header("Детекция (YOLO)")
    if not _YOLO_OK:
        st.error("Пакет ultralytics не установлен. Установите: pip install ultralytics")
        return

    with st.sidebar:
        st.subheader("Выбор модели")
        st.selectbox("Выберите модель YOLO", ["YOLOv8"], index=0)
        weights_path = st.text_input("Путь к модели YOLOv8", value="models/YOLOv8.pt")

        st.subheader("Опции процессора")
        device = st.radio("Выберите устройство", ["CPU", "GPU"], index=0)
        device_arg = "cpu" if device == "CPU" else 0

        st.subheader("Настройки детекции")
        conf = st.slider("Порог уверенности", 0.0, 1.0, 0.25, 0.01)
        iou  = st.slider("IoU", 0.1, 0.9, 0.50, 0.05)
        thick = st.slider("Толщина линии", 1, 10, 2, 1)

    tabs = st.tabs(["Изображение (JPG/PNG)", "Видео (локальный файл)"])

    with tabs[0]:
        file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
        if file:
            img_bytes = file.read()
            st.image(img_bytes, caption=file.name, width=360)
            if st.button("Запустить детекцию на изображении", type="primary"):
                if not Path(weights_path).exists():
                    st.error("Файл весов не найден.")
                else:
                    model = YOLO(weights_path); model.to(device_arg)
                    res = _run_yolo_on_image(model, img_bytes, conf=conf, iou=iou, thickness=thick)
                    st.subheader(f"Найдено объектов: {res['n']}")
                    buf = io.BytesIO(); res["annotated"].save(buf, format="PNG")
                    st.image(buf.getvalue(), caption="Результат детекции", width=480)
                    if res["top"] is not None:
                        insert_detection(
                            db_path, ts=dt.datetime.now().isoformat(timespec="seconds"),
                            kind="yolo", source="image", filename=file.name, model=Path(weights_path).name,
                            label_en=res["top"]["label_en"], label_ru=res["top"]["label_en"], score=float(res["top"]["score"]),
                            json_top5=None, price_min=None, price_max=None, currency=None, image_path=None,
                            plate_number=None, region=None, year_issued=None, brand=None, car_type=None, color=None, conf_avg=None, extra_json=None
                        )
                        st.toast("Сохранено в detections.db", icon="💾")

    with tabs[1]:
        vfile = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi", "mkv"])
        c1, c2, c3 = st.columns(3)
        with c1: display_width = st.slider("Ширина окна, px", 320, 1280, 720, 10)
        with c2: infer_every_n_frames = st.slider("Инференс на каждом N-м кадре", 1, 10, 2, 1)
        with c3: save_every_sec = st.slider("Сохранять в БД раз в, сек", 1, 60, 10, 1)
        write_annotated = st.checkbox("Записывать аннотированное видео (MP4)", value=False)
        start = st.button("▶️ Начать воспроизведение", type="primary"); stop = st.button("⏹ Остановить")
        if start: st.session_state["video_running"] = True
        if stop:  st.session_state["video_running"] = False
        if vfile and st.session_state.get("video_running", False):
            if not Path(weights_path).exists():
                st.error("Файл весов не найден."); st.session_state["video_running"] = False
            else:
                try: model = YOLO(weights_path)
                except Exception as e:
                    st.error(f"Не удалось загрузить модель: {e}"); st.session_state["video_running"] = False
                else:
                    _play_local_video_with_yolo(
                        db_path=db_path, model=model, video_bytes=vfile.read(), filename=vfile.name,
                        conf=conf, iou=iou, device_arg=("cpu" if device == "CPU" else 0),
                        display_width=int(display_width), infer_every_n_frames=int(infer_every_n_frames),
                        save_every_sec=float(save_every_sec), write_annotated=bool(write_annotated), thickness=int(thick),
                    )

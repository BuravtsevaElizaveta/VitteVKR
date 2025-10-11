# components/plate_assistant.py
# Номер + ИИ-ассистент:
# - Детекция пластины: YOLO (без ресайза) → Haar (без ресайза) → морфологический локатор (fallback).
# - Если TensorFlow доступен: сегментация символов + CNN (model.h5).
# - Если TensorFlow недоступен: извлечение номера через лёгкую нормализацию + LLM-визион.
# - Строка номера выводится всегда (даже если символы распознаны как '?').

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional
import re, json, base64, time, logging
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

# ── TensorFlow: опционально
TF_OK = True
TF_ERR = None
try:
    import tensorflow as tf  # может отсутствовать в облаке (Py 3.13)
except Exception as _e:
    TF_OK = False
    TF_ERR = _e

from utils.db import insert_detection
from openai_config import get_client, get_defaults

# ── YOLO (опционально)
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    _YOLO_OK = False

log = logging.getLogger(__name__)

# Алфавит модели (legacy CNN)
_CHARS = '0123456789АВЕКМНОРСТУХ'
_NUM2CHAR = {i: ch for i, ch in enumerate(_CHARS)}

# ── Нормализация номера без TF (латиница↔кириллица, фильтры)
_MAP = str.maketrans({
    "O": "0", "o": "0", "I": "1", "l": "1",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X"
})
PLATE_RE = re.compile(r"[A-ZА-Я0-9]{5,10}")

def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().upper().translate(_MAP)
    m = PLATE_RE.findall(t)
    return m[0] if m else t

# ── ProxyAPI helpers
def _extract_text(resp) -> str:
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        if hasattr(resp, "output"):
            parts = []
            for itm in resp.output:
                if getattr(itm, "type", None) == "message":
                    for c in getattr(itm, "content", []):
                        t = getattr(c, "text", None)
                        if t:
                            parts.append(t)
            if parts:
                return " ".join(parts).strip()
    except Exception:
        pass
    return ""

def _img_to_b64(img_bgr: np.ndarray, max_side: int = 224, q: int = 35) -> str:
    h, w = img_bgr.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return base64.b64encode(enc).decode() if ok else ""

def _gpt_vision(client, sys_prompt: str, user_text: str, img_bgr: Optional[np.ndarray], model: str) -> str:
    content = [{"type": "input_text", "text": user_text}]
    if img_bgr is not None:
        content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + _img_to_b64(img_bgr)})
    try:
        r = client.responses.create(
            model=model,
            input=[{"role": "system", "content": sys_prompt},
                   {"role": "user", "content": content}],
            temperature=0.0, max_output_tokens=200
        )
        return _extract_text(r)
    except Exception as e:
        log.warning("Proxy error: %s", e)
        return ""

def gpt_brand(img, c, m): return _gpt_vision(c, "Определи марку автомобиля. Верни только марку.", "Какая марка авто?", img, m) or "Не удалось распознать марку"
def gpt_color(img, c, m): return _gpt_vision(c, "Определи основной цвет кузова (одно слово по-русски).", "Какой цвет авто?", img, m) or "Не удалось распознать цвет"
def gpt_type (img, c, m): return _gpt_vision(c, "Классифицируй тип: легковой/грузовой/автобус/мотоцикл. Верни одно слово.", "Какой тип ТС?", img, m) or "Не удалось классифицировать тип автомобиля"

def gpt_plate(img, c, m) -> str:
    # Чтение номера как текста: коротко и строго
    sys = ("Ты распознаёшь государственный номер на фотографии. Верни ТОЛЬКО номер без комментариев. "
           "Допускается формат: А123ВС77/А123ВС777 (кириллица) или аналог латиницей.")
    txt = _gpt_vision(c, sys, "Считай номер и верни только его.", img, m)
    return normalize_plate_text(txt)

def gpt_reg_year(plate, c, m):
    if not plate:
        return "", ""
    sys = ("Ты анализируешь российский госномер (А123ВС77/А123ВС777). "
           "По коду региона назови субъект РФ и предположи год выдачи. "
           "Строго JSON: {\"region_name\":\"...\",\"year_issued\":\"...\"}.")
    txt = _gpt_vision(c, sys, f"Определи регион+год выдачи для: {plate}", None, m)
    try:
        j = json.loads(txt); return j.get("region_name", ""), j.get("year_issued", "")
    except Exception:
        return "", ""

# ── Поиск файлов
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def _find_yolo(user: Optional[Path]) -> Optional[Path]:
    return _first_existing([user or Path(), Path("models/YOLOv8.pt"), Path("utils/YOLOv8.pt"), Path("YOLOv8.pt")])

def _find_cascade(user: Optional[Path]) -> Optional[Path]:
    return _first_existing([
        user or Path(),
        Path("haarcascade_licence_plate_rus_16stages.xml"),
        Path("haarcascade_russian_plate_number.xml"),
        Path("models/cascades/haarcascade_licence_plate_rus_16stages.xml"),
        Path("models/haarcascade_licence_plate_rus_16stages.xml"),
    ])

# ── Кэш моделей
@st.cache_resource(show_spinner="Загрузка CNN…")
def _load_cnn(path: Path):
    if not TF_OK:
        return None
    return tf.keras.models.load_model(str(path))

@st.cache_resource(show_spinner="Загрузка YOLO…")
def _load_yolo(path: Path):
    if not _YOLO_OK:
        return None
    try:
        return YOLO(str(path))
    except Exception as e:
        log.error("YOLO load failed: %s", e)
        return None

# ── Геометрия окна
def _is_plausible_box(x, y, w, h, H, W) -> bool:
    if w <= 0 or h <= 0: return False
    ar = w / float(h)
    h_frac, w_frac = h/float(H), w/float(W)
    # госномер: широкий (2.2–6.8), по высоте 3–20% кадра, по ширине 8–60%
    if not (2.2 <= ar <= 6.8): return False
    if not (0.03 <= h_frac <= 0.20): return False
    if not (0.08 <= w_frac <= 0.60): return False
    return True

def _edge_density(gray_roi: np.ndarray) -> float:
    e = cv2.Canny(gray_roi, 70, 160)
    return float(e.mean()) / 255.0

# ── Морфологический локатор (fallback/уточнение)
def _locate_plate_morph(img_bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)
    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    grad = np.absolute(grad)
    grad = (255 * (grad - grad.min()) / (grad.ptp() + 1e-6)).astype(np.uint8)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_k, iterations=1)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rect_k, iterations=2)
    bw = cv2.medianBlur(bw, 3)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    best = None; best_score = -1.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not _is_plausible_box(x, y, w, h, H, W):
            continue
        roi = gray[y:y+h, x:x+w]
        score = 0.6 * _edge_density(roi) + 0.4 * (1.0 - min(1.0, abs((w/h) - 4.5) / 3.5))
        if score > best_score:
            best_score = score; best = (x, y, w, h)
    return best

# ── Haar: без ресайза + строгая фильтрация
def plate_detect_haar(img_bgr: np.ndarray, cascade_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        st.warning("Каскад не загрузился. Проверьте путь к *.xml.")
        return img_bgr, None

    rects = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4,
                                     minSize=(50, 16), flags=cv2.CASCADE_SCALE_IMAGE)
    H, W = gray.shape[:2]
    best = None; best_score = -1.0
    for (x, y, w, h) in rects:
        if not _is_plausible_box(x, y, w, h, H, W):
            continue
        roi = gray[y:y+h, x:x+w]
        score = 0.6 * _edge_density(roi) + 0.4 * (1.0 - min(1.0, abs((w/h) - 4.5) / 3.5))
        if score > best_score:
            best_score = score; best = (x, y, w, h)

    if best is None:
        best = _locate_plate_morph(img_bgr)
        if best is None:
            return img_bgr, None

    x, y, w, h = best
    pad_x, pad_y = int(0.03*w), int(0.12*h)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)

    roi = img_bgr[y1:y2, x1:x2].copy()
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out, roi

# ── YOLO: без ресайза входа
def plate_detect_yolo(img_bgr: np.ndarray, model) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    res = model.predict(img_bgr, verbose=False, device="cpu", conf=0.25, iou=0.5)
    H, W = img_bgr.shape[:2]
    for r in res:
        names = getattr(r, "names", {}) or {}
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)
        cand = [i for i, c in enumerate(cls) if str(names.get(int(c), "")).lower().replace("-", "_") in ("license_plate", "licenseplate", "plate")]
        if not cand:
            ar = (xyxy[:, 2] - xyxy[:, 0]) / np.maximum(1, (xyxy[:, 3] - xyxy[:, 1]))
            scores = conf + (np.clip(1.0 - np.abs(ar - 4.5) / 3.5, 0, 1) * 0.5)
            idx = int(np.argmax(scores))
        else:
            idx = int(cand[np.argmax(conf[cand])])

        x1, y1, x2, y2 = xyxy[idx]
        w, h = x2 - x1, y2 - y1
        if not _is_plausible_box(x1, y1, w, h, H, W):
            best = _locate_plate_morph(img_bgr)
            if best is not None:
                x, y, w, h = best
                x1, y1, x2, y2 = x, y, x + w, y + h

        pad_x, pad_y = int(0.03*(x2-x1)), int(0.12*(y2-y1))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(W, x2 + pad_x), min(H, y2 + pad_y)

        roi = img_bgr[y1:y2, x1:x2].copy()
        out = img_bgr.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return out, roi
    return img_bgr, None

def plate_detect(img_bgr: np.ndarray, method: str, yolo_path: Optional[Path], cascade_path: Optional[Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if method == "YOLOv8":
        ypath = _find_yolo(yolo_path)
        if ypath:
            y = _load_yolo(ypath)
            if y is not None:
                out, roi = plate_detect_yolo(img_bgr, y)
                if roi is not None:
                    return out, roi
    cpath = _find_cascade(cascade_path)
    if cpath:
        return plate_detect_haar(img_bgr, cpath)
    st.warning("Не найден файл каскада для Haar-детекции.")
    return img_bgr, None

# ── Сегментация символов (legacy pipeline для CNN)
def _segment_characters_legacy(image: np.ndarray) -> np.ndarray:
    """
    Возвращает массив 2D-патчей (каждый 44x24, белые символы на чёрном фоне),
    отсортированных слева направо. Это именно тот препроцесс, на котором обучалась model.h5.
    """
    if image is None:
        return np.array([])

    img_lp = cv2.resize(image, (333, 75))  # нормализация размера таблички
    img_gray = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    # Адаптивная бинаризация (фон белый, символы чёрные)
    img_bin = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    # Склеиваем разрывы
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # Чистим края
    img_bin[0:3, :], img_bin[:, 0:3] = 255, 255
    img_bin[72:75, :], img_bin[:, 330:333] = 255, 255

    LP_W, LP_H = img_bin.shape
    dims = [LP_W / 6, LP_W / 2, LP_H / 10, 2 * LP_H / 3]
    lower_h, upper_h = dims[0] * 0.5, dims[1] * 1.2
    lower_w, upper_w = dims[2] * 0.5, dims[3] * 1.2

    contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return np.array([])

    chars, xs = [], []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue  # внешний контур
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h)
        if lower_h < h < upper_h and lower_w < w < upper_w and 0.3 < ar < 1.2:
            # Инверсия символа → белый символ на чёрном (как ждёт CNN)
            char = cv2.subtract(255, cv2.resize(img_bin[y:y+h, x:x+w], (20, 40)))
            # Паддинг до 44x24
            char_copy = np.zeros((44, 24), dtype=np.uint8)
            char_copy[2:42, 2:22] = char
            chars.append(char_copy)
            xs.append(x)

    if not chars:
        return np.array([])

    idx_sorted = np.argsort(np.array(xs))
    return np.array([chars[i] for i in idx_sorted])

def _fix_dim(img):
    return np.stack((img,)*3, axis=-1) if img.ndim == 2 else img

def recognize_number_cnn(roi: np.ndarray, model, thr: float) -> Tuple[str, List[float]]:
    if model is None:
        return "", []
    seg_chars = _segment_characters_legacy(roi)
    if seg_chars.size == 0:
        return "", []
    out, confs = [], []
    for ch in seg_chars:
        img28 = cv2.resize(ch, (28, 28))
        X = _fix_dim(img28) / 255.0
        pr = model.predict(X.reshape(1, 28, 28, 3), verbose=0)
        idx = int(np.argmax(pr))
        cf = float(pr[0, idx])
        out.append(_NUM2CHAR[idx] if cf >= thr else "?")
        confs.append(cf)
    return "".join(out), confs

def fix_number_format_new_rus(s: str, confs: List[float]) -> Tuple[str, List[float]]:
    pat = r"^([АВЕКМНОРСТУХ])\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$"
    if re.match(pat, s or ""): return s, confs
    if s and s[-1] in _CHARS:
        cand = s[:-1]
        if re.match(pat, cand or ""): return cand, confs[:-1]
    return s or "", confs

# ── UI
def render_plate_assistant(db_path: Path):
    st.header("🪪 Номер + ИИ-ассистент")

    defaults = get_defaults()
    with st.sidebar:
        st.subheader("ProxyAPI")
        api_key = st.text_input("API key", value=defaults["api_key"], type="password")
        base_url = st.text_input("Base URL", value=defaults["base_url"])
        gpt_model = st.selectbox("Модель", ["gpt-4o-mini","gpt-4o","gpt-4o-mini-compact"], index=0)

        st.subheader("Модели")
        yolo_path_ui = st.text_input("YOLOv8 веса (ищем также utils/YOLOv8.pt)", value="models/YOLOv8.pt")
        cascade_path_ui = st.text_input("Haar Cascade (автопоиск)", value="haarcascade_licence_plate_rus_16stages.xml")
        cnn_path = st.text_input("CNN (model.h5)", value="model.h5")

        st.subheader("Параметры")
        fmt = st.selectbox("Формат номера", ['Старые РФ номера','Новые РФ номера','Зарубежные номера'], index=1)
        det_m = st.selectbox("Метод детекции номера", ['YOLOv8','Haar Cascade (legacy)'],
                             index=0 if (_find_yolo(Path(yolo_path_ui)) is not None and _YOLO_OK) else 1)
        thr = st.slider("Порог уверенности (CNN), ниже — '?'", 0.0, 1.0, 0.50, 0.05)

    if not TF_OK:
        st.info("TensorFlow недоступен в этом деплое (Py 3.13). Будет использован fallback: лёгкая нормализация + LLM-визион.")
        with st.expander("Техническая деталь"):
            st.code(repr(TF_ERR))

    c1, c2 = st.columns([1, 1])
    with c1:
        f = st.file_uploader("Изображение автомобиля", type=["jpg","jpeg","png"])
    with c2:
        do_all = st.checkbox("Выполнить всё (марка/тип/цвет/регион)", value=True)
        btn = st.button("Анализировать", type="primary", use_container_width=True)

    if f:
        pil = Image.open(f)
        w = 420
        st.image(pil.resize((w, int(pil.height*(w/pil.width))), Image.Resampling.LANCZOS),
                 caption="Загруженное изображение (уменьшено)", width=420)

    if not btn or not f:
        return

    # Готовим клиента LLM заранее (нужен и для TF-fallback)
    client = get_client(api_key=api_key, base_url=base_url)
    img_bgr = np.array(Image.open(f).convert("RGB"))[:, :, ::-1]  # полное разрешение

    # 1) Детекция пластины
    det_img, roi = plate_detect(img_bgr, det_m, Path(yolo_path_ui), Path(cascade_path_ui))
    if roi is None:
        best = _locate_plate_morph(img_bgr)
        if best is not None:
            x,y,w,h = best
            pad_x, pad_y = int(0.03*w), int(0.12*h)
            x1,y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2,y2 = min(img_bgr.shape[1], x+w+pad_x), min(img_bgr.shape[0], y+h+pad_y)
            roi = img_bgr[y1:y2, x1:x2].copy()
            det_img = img_bgr.copy()
            cv2.rectangle(det_img, (x1,y1), (x2,y2), (0,255,0), 2)

    st.subheader("Распознавание номера…")
    st.image(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB), caption="Обнаруженный номер", width=360)

    # 2) Распознавание: CNN (если доступен) → иначе LLM-визион
    plate, confs = "", []
    if TF_OK:
        try:
            cnn = _load_cnn(Path(cnn_path))
        except Exception as e:
            cnn = None
            st.warning(f"CNN не загрузился, используем LLM-fallback: {e}")
        plate, confs = recognize_number_cnn(roi, cnn, thr) if cnn is not None else ("", [])
    if not plate:
        # fallback: LLM читает номер как текст
        plate = gpt_plate(roi if roi is not None else img_bgr, client, gpt_model)
        confs = []  # у LLM нет покадровых уверенности
    if fmt == 'Новые РФ номера' and plate:
        plate, confs = fix_number_format_new_rus(plate, confs)

    # 3) Доп. атрибуты через LLM (по желанию)
    brand = color = car_type = ""
    if do_all:
        brand = gpt_brand(img_bgr, client, gpt_model)
        color = gpt_color(img_bgr, client, gpt_model)
        car_type = gpt_type(img_bgr, client, gpt_model)
    region, year = ("","")
    if plate and fmt in ('Старые РФ номера','Новые РФ номера'):
        st.subheader("Дополнительный анализ номера (регион, год выдачи)…")
        region, year = gpt_reg_year(plate, client, gpt_model)

    # 4) Вывод (номер выводим ВСЕГДА)
    st.success("Готово.")
    st.header("Результаты анализа")
    st.write(f"**Номер**: {plate or '—'}")
    if region: st.write(f"**Регион**: {region}")
    if year:   st.write(f"**Год выдачи**: {year}")
    if confs:
        avg = float(sum(confs)/len(confs))
        st.write(f"Средняя уверенность (CNN): {avg:.2f}")
        if plate and len(plate) == len(confs):
            df = pd.DataFrame({"Символ": list(plate), "Уверенность": confs})
            st.bar_chart(df, x="Символ", y="Уверенность")
    if brand:    st.write(f"**Марка**: {brand}")
    if car_type: st.write(f"**Тип**: {car_type}")
    if color:    st.write(f"**Цвет**: {color}")

    # 5) БД
    avg_conf = float(sum(confs)/len(confs)) if confs else None
    insert_detection(
        db_path=db_path,
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        kind="plate", source="image", filename=f.name,
        model=("YOLOv8→" if det_m=="YOLOv8" else "") + ("Haar/Morph+CNN" if TF_OK else "Haar/Morph+LLM"),
        label_en=plate or None, label_ru=plate or None, score=avg_conf,
        json_top5=None, price_min=None, price_max=None, currency=None, image_path=None,
        plate_number=plate or None, region=region or None, year_issued=year or None,
        brand=brand or None, car_type=car_type or None, color=color or None, conf_avg=avg_conf, extra_json=None
    )
    st.toast("Сохранено в detections.db", icon="💾")

# components/parts_assistant.py
# Вкладка «Запчасть + ИИ-ассистент (ProxyAPI)».

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import io, json, base64, datetime as dt, logging

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --- локальный utils раньше внешних (во избежание коллизии с cv2.utils) ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from openai_config import get_client, get_defaults
from utils.db import insert_detection
from utils.pricing import load_price_map, lookup_price, fallback_price

# YOLO опционально (если нет ultralytics — вкладка всё равно работает)
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    _YOLO_OK = False


def _img_bytes_to_data_uri(img_bytes: bytes, max_side: int = 768, quality: int = 80) -> str:
    """Сжать изображение и вернуть data: URI для GPT-визии."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64


def _safe_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_text(resp) -> str:
    """Достаём текст из ProxyAPI/OpenAI v1 responses.create()."""
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


DEFAULT_LABELS = [
    "alternator","battery","brake disc","brake pad","bumper","car door","clutch","crankshaft",
    "cylinder head","drive shaft","engine","exhaust pipe","fan belt","fender","fuel injector",
    "gearbox","headlight","hood","ignition coil","mirror","muffler","oil filter","piston",
    "radiator","rim","seat","shock absorber","spark plug","spring","steering wheel",
    "tail light","throttle body","tire","turbocharger","valve","water pump","wheel",
    "windscreen","wiper","air filter","catalytic converter","control arm","door handle",
    "camshaft","oxygen sensor","strut","bearing","axle","fuel pump"
][:50]

RU_SEED = {
    "alternator":"генератор","battery":"аккумулятор","brake disc":"тормозной диск","brake pad":"тормозная колодка",
    "bumper":"бампер","car door":"дверь автомобиля","clutch":"сцепление","crankshaft":"коленвал","cylinder head":"головка блока цилиндров",
    "drive shaft":"карданный вал","engine":"двигатель","exhaust pipe":"выхлопная труба","fan belt":"ремень вентилятора","fender":"крыло",
    "fuel injector":"топливная форсунка","gearbox":"коробка передач","headlight":"фара","hood":"капот","ignition coil":"катушка зажигания",
    "mirror":"зеркало","muffler":"глушитель","oil filter":"масляный фильтр","piston":"поршень","radiator":"радиатор",
    "rim":"диск колеса","seat":"сиденье","shock absorber":"амортизатор","spark plug":"свеча зажигания","spring":"пружина",
    "steering wheel":"руль","tail light":"задний фонарь","throttle body":"дроссельный узел","tire":"шина",
    "turbocharger":"турбокомпрессор","valve":"клапан","water pump":"водяной насос","wheel":"колесо","windscreen":"лобовое стекло",
    "wiper":"дворник","air filter":"воздушный фильтр","catalytic converter":"каталитический нейтрализатор",
    "control arm":"рычаг подвески","door handle":"дверная ручка","camshaft":"распредвал","oxygen sensor":"датчик кислорода (лямбда-зонд)",
    "strut":"стойка подвески","bearing":"подшипник","axle":"ось","fuel pump":"топливный насос"
}


def _proxy_classify(client: OpenAI, img_bytes: bytes, labels: List[str], model: str) -> List[Tuple[str, float]]:
    """Классифицируем фото детали среди labels через GPT-визию ProxyAPI (softmax-подобный ответ)."""
    uri = _img_bytes_to_data_uri(img_bytes, max_side=768, quality=80)
    sys_msg = (
        "Ты — CV-ассистент. По фото автозапчасти верни JSON строго формата:\n"
        "{\"top\": [{\"label\": <строка из списка>, \"score\": <число 0..1>}, ...]}"
    )
    user = [
        {"type": "input_text", "text": "Варианты (выбери строго из них): " + ", ".join(labels)},
        {"type": "input_image", "image_url": uri},
    ]
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
            temperature=0.0,
            max_output_tokens=200,
        )
        data = _safe_json(_extract_text(resp))
        if data and isinstance(data.get("top"), list) and data["top"]:
            pairs: List[Tuple[str, float]] = []
            for it in data["top"]:
                lab = str(it.get("label", "")).strip()
                sc = float(it.get("score", 0.0))
                if lab in labels:
                    pairs.append((lab, sc))
            if pairs:
                s = sum(max(0.0, x[1]) for x in pairs) or 1.0
                # нормализуем «оценки» в вероятности
                return [(lab, float(sc) / s) for lab, sc in pairs][:5]
    except Exception as e:
        logging.error(f"[Proxy classify] {e}")
    # fallback
    return [(labels[0], 1.0)]


def _proxy_translate(client: OpenAI, labels_en: List[str], model: str) -> Dict[str, str]:
    """Быстрый перевод меток EN→RU (с seed-словарём)."""
    res = {k: RU_SEED.get(k, k) for k in labels_en}
    pending = [l for l in labels_en if l not in RU_SEED]
    if not pending:
        return res
    sys_msg = (
        "Ты переводчик технических терминов автозапчастей. Верни JSON строго формата:\n"
        "{\"map\": {\"english\": \"русский\"}}"
    )
    user = [{"type": "input_text", "text": "Переведи: " + ", ".join(pending)}]
    try:
        r = client.responses.create(
            model=model,
            input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
            temperature=0.0,
            max_output_tokens=200,
        )
        data = _safe_json(_extract_text(r)) or {}
        if isinstance(data.get("map"), dict):
            for k, v in data["map"].items():
                if k in pending and v:
                    res[k] = str(v).strip()
    except Exception as e:
        logging.error(f"[Proxy translate] {e}")
    return res


def _yolo_crops(
    img_bytes: bytes,
    weights_path: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    max_crops: int = 8,
) -> List[bytes]:
    """Если YOLO доступен — сначала вырезаем деталь(и) и классифицируем их отдельно."""
    if not _YOLO_OK or not Path(weights_path).exists():
        return []
    model = YOLO(str(weights_path))
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    res = model.predict(source=np.array(img), conf=conf, iou=iou, verbose=False)
    crops: List[bytes] = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes.xyxy.cpu().numpy().astype(int)[:max_crops]:
            x1, y1, x2, y2 = [max(0, v) for v in b]
            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            crops.append(buf.getvalue())
    return crops


def render_parts_assistant(
    db_path: Path,
    default_label_map_path: Path | None = None,
    default_price_map_path: Path | None = None,
):
    st.header("🧠 Запчасть + ИИ-ассистент (ProxyAPI)")

    defaults = get_defaults()
    with st.sidebar:
        st.subheader("ProxyAPI")
        api_key = st.text_input("API key", value=defaults["api_key"], type="password")
        base_url = st.text_input("Base URL", value=defaults["base_url"])
        model = st.selectbox("Модель", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-compact"], index=0)

        st.subheader("Опции распознавания")
        use_yolo = st.checkbox("Сначала выделять детали (YOLO)", value=False)
        yolo_path = st.text_input("Путь к YOLO-весам", value="Приложение/models/YOLOv8.pt")
        max_crops = st.slider("Макс. фрагментов", 1, 12, 6)

        st.subheader("Классы")
        labels = DEFAULT_LABELS.copy()
        if default_label_map_path and default_label_map_path.exists():
            try:
                lm = json.loads(default_label_map_path.read_text("utf-8"))
                if isinstance(lm, dict):
                    labels = [
                        v for _, v in sorted(
                            ((int(k), str(v)) for k, v in lm.items()),
                            key=lambda x: x[0]
                        )
                    ]
                elif isinstance(lm, list):
                    labels = [str(x) for x in lm]
            except Exception:
                pass
        labels = [
            ln.strip()
            for ln in st.text_area(
                "Список меток (по одной в строке)", value="\n".join(labels), height=220
            ).splitlines()
            if ln.strip()
        ]
        st.caption(f"Итого меток: **{len(labels)}**")

        price_map = load_price_map(default_price_map_path)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 1) Загрузите изображение детали")
        file = st.file_uploader("JPG/PNG", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        demo_btn = st.button("Демо-картинка", use_container_width=True)
    with c2:
        st.markdown("### 2) Распознавание")
        run_btn = st.button(
            "🔍 Определить запчасть",
            type="primary",
            use_container_width=True,
            disabled=not (file or demo_btn),
        )

    img_bytes: Optional[bytes] = None
    filename: Optional[str] = None
    if file:
        img_bytes = file.read()
        filename = file.name
    elif demo_btn:
        demo_path = Path("ex2.jpg") if Path("ex2.jpg").exists() else Path("ex.png")
        if demo_path.exists():
            img_bytes = demo_path.read_bytes()
            filename = demo_path.name
        else:
            st.warning("Демо-файл не найден (ex2.jpg / ex.png)")

    if img_bytes:
        st.image(img_bytes, caption=filename or "изображение", width=360)

    if run_btn and img_bytes:
        client = get_client(api_key=api_key, base_url=base_url)

        # 1) Опциональный YOLO pre-crop
        crops = _yolo_crops(img_bytes, Path(yolo_path), max_crops=max_crops) if use_yolo else []
        if not crops:
            crops = [img_bytes]

        # 2) Классификация всех кропов + перевод + оценка цены
        results_all: List[Dict[str, Any]] = []
        prog = st.progress(0)
        for i, crop in enumerate(crops):
            pairs = _proxy_classify(client, crop, labels, model)
            labs = [lab for lab, _ in pairs]
            ru_map = _proxy_translate(client, labs, model)
            top_en, top_sc = pairs[0]
            price = lookup_price(top_en, price_map) or fallback_price(top_en)

            results_all.append(
                dict(top=pairs, ru_map=ru_map, price=price, crop_index=i)
            )
            prog.progress(int((i + 1) / len(crops) * 100))

        st.success("Готово!")

        # 3) Вывод по каждому кропу
        tabs = st.tabs([f"Фрагмент {i+1}" for i in range(len(results_all))])
        for tab, res in zip(tabs, results_all):
            with tab:
                if len(crops) > 1:
                    st.image(
                        crops[res["crop_index"]],
                        caption=f"crop #{res['crop_index']+1}",
                        width=360,
                    )
                df = pd.DataFrame(
                    [
                        {
                            "rank": j + 1,
                            "label_en": lab,
                            "label_ru": res["ru_map"].get(lab, lab),
                            "score": sc,
                        }
                        for j, (lab, sc) in enumerate(res["top"])
                    ]
                )
                st.dataframe(df, use_container_width=True)

                price = res["price"]
                st.markdown(
                    f"**Цена (ориентир):** {price['price_min']}–{price['price_max']} {price['currency']} · "
                    f"**Товар:** {res['ru_map'].get(res['top'][0][0], res['top'][0][0])}"
                )

        # 4) Сохранение лучшего результата в БД
        best = results_all[0]
        top_en, top_sc = best["top"][0]
        top_ru = best["ru_map"].get(top_en, top_en)
        price = best["price"]

        insert_detection(
            db_path=db_path,
            ts=dt.datetime.now().isoformat(timespec="seconds"),
            kind="parts",
            source="image",
            filename=filename or "uploaded.jpg",
            model=model,
            label_en=top_en,
            label_ru=top_ru,
            score=float(top_sc),
            json_top5=json.dumps(
                [
                    {
                        "label_en": lab,
                        "label_ru": best["ru_map"].get(lab, lab),
                        "score": sc,
                    }
                    for lab, sc in best["top"]
                ],
                ensure_ascii=False,
            ),
            price_min=float(price["price_min"]),
            price_max=float(price["price_max"]),
            currency=price["currency"],
            image_path=None,
            plate_number=None,
            region=None,
            year_issued=None,
            brand=None,
            car_type=None,
            color=None,
            conf_avg=None,
            extra_json=None,
        )
        st.toast("Сохранено в detections.db", icon="💾")


# utils/pricing.py — прайс-логика: загрузка price_map.json и разумные фоллбэки.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json

def load_price_map(path: Path | None) -> Dict[str, Dict]:
    if not path or not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text("utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def lookup_price(label_en: str, price_map: Dict[str, Dict]) -> Optional[Dict]:
    item = price_map.get(label_en)
    if not item:
        return None
    return dict(
        price_min=float(item.get("price_min", 0)),
        price_max=float(item.get("price_max", 0)),
        currency=str(item.get("currency", "RUB"))
    )

def fallback_price(label_en: str) -> Dict:
    rnd = abs(hash(label_en)) % 10_000
    base = 1500 + (rnd % 7000)   # 1500..8499
    spread = 0.6 + (rnd % 400) / 1000.0
    pmin = int(base)
    pmax = int(base * (1.0 + spread))
    return dict(price_min=pmin, price_max=pmax, currency="RUB")

"""
openai_config.py — конфигурация ProxyAPI/OpenAI SDK v1.* для приложения.
"""

import os
from typing import Optional
from openai import OpenAI

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-gpTBIn8Bs3JNpcZcsorUMegcZVKEVQy0")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
DEFAULT_MODEL   = os.getenv("OPENAI_MODEL",   "gpt-4o-mini")

def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    key = api_key or DEFAULT_API_KEY
    base = base_url or DEFAULT_BASE_URL
    return OpenAI(api_key=key, base_url=base)

def get_defaults():
    return dict(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL)

"""Настройки сервиса. Значения берутся из переменных окружения или файла .env."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Переменные, заданные в самом окружении, важнее значений из .env (override=False).
load_dotenv(PROJECT_ROOT / ".env")

# Самая дешёвая актуальная модель OpenAI для массовых задач (ноябрь 2026).
# gpt-4o-mini и gpt-4.1-mini отключаются 14.01.2027, поэтому на них не опираемся.
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "low"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    reasoning_effort: str | None
    temperature: float | None
    request_timeout: float
    max_repair_attempts: int


def is_placeholder_key(api_key: str) -> bool:
    """В .env.example стоит «ваш_ключ». Кириллица в заголовке Authorization уронила бы
    HTTP-клиент с невнятной ошибкой кодировки, поэтому ловим заглушку заранее."""
    return not api_key.isascii() or api_key.lower() in {"your_key", "your_api_key", "sk-..."}


def _read_env(name: str, default: str | None = None) -> str | None:
    """Переменная не задана: берём default. Задана пустой строкой: значит «не передавать»."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip() or None


def _read_number(name: str, default, cast=float):
    raw_value = _read_env(name)
    if raw_value is None:
        return default
    try:
        return cast(raw_value)
    except ValueError:
        raise RuntimeError(f"Переменная {name} должна быть числом, сейчас в ней {raw_value!r}") from None


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=_read_env("OPENAI_API_KEY") or "",
        openai_model=_read_env("OPENAI_MODEL") or DEFAULT_MODEL,
        openai_base_url=_read_env("OPENAI_BASE_URL"),
        reasoning_effort=_read_env("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        temperature=_read_number("OPENAI_TEMPERATURE", None),
        request_timeout=_read_number("OPENAI_TIMEOUT", 60.0),
        max_repair_attempts=max(0, _read_number("MAX_REPAIR_ATTEMPTS", 2, cast=int)),
    )

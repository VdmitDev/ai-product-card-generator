"""HTTP API и веб-форма.

Запуск: uvicorn app.main:app --reload
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from app.config import PROJECT_ROOT, get_settings, is_placeholder_key
from app.llm_service import LLMServiceError, get_generator
from app.schemas import GenerationResult, ProductInput, format_validation_errors

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
# Примеры ошибок для кнопок на странице, в порядке показа. У товаров подпись берётся из name
EDGE_CASE_TITLES = {
    "incomplete_data": "Неполные данные",
    "invalid_json": "Битый JSON",
    "empty_name": "Пустое название",
}

ERROR_RESPONSES = {
    422: {"description": "Некорректный JSON или данные не прошли проверку"},
    500: {"description": "Сервис не настроен: нет ключа, неверный ключ или модель"},
    502: {"description": "OpenAI вернул ошибку или ответ не по схеме"},
    503: {"description": "OpenAI недоступен, превышен лимит или закончилась квота"},
    504: {"description": "OpenAI не ответил вовремя"},
}

app = FastAPI(
    title="Генератор карточек товаров КИП",
    description=(
        "Черновики карточек для B2B-магазина: SEO-заголовок и описание, характеристики, "
        "поисковые запросы и памятка для менеджера. Технические значения берутся только из входных данных."
    ),
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    is_broken_json = any(error.get("type") == "json_invalid" for error in errors)
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_json" if is_broken_json else "validation_error",
            "message": (
                "Тело запроса не является корректным JSON."
                if is_broken_json
                else "Входные данные не прошли проверку."
            ),
            "details": format_validation_errors(errors, raw_body=exc.body),
        },
    )


@app.exception_handler(LLMServiceError)
async def handle_llm_error(request: Request, exc: LLMServiceError) -> JSONResponse:
    logger.warning("Генерация не удалась [%s]: %s", exc.error_code, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.error_code, "message": exc.message})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Необработанная ошибка при запросе %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Внутренняя ошибка сервиса, подробности в логе."},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    # Файл читаем на каждый запрос: страница маленькая, зато правки видны без перезапуска
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.post("/generate", response_model=GenerationResult, responses=ERROR_RESPONSES)
def generate(
    product: ProductInput,
    output_format: Literal["json", "markdown"] = Query(
        "json", alias="format", description="json: полный ответ; markdown: только карточка текстом"
    ),
):
    # Обычная def, а не async: FastAPI выполнит её в пуле потоков, и синхронный
    # клиент OpenAI не будет блокировать остальные запросы.
    # Генератор берём здесь, а не через Depends: зависимости FastAPI вычисляет до
    # проверки тела, и без ключа пользователь не увидел бы ошибок в своём JSON.
    result = get_generator().generate(product)
    if output_format == "markdown":
        return PlainTextResponse(result.markdown, media_type="text/markdown; charset=utf-8")
    return result


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    key = settings.openai_api_key
    return {
        "status": "ok",
        "model": settings.openai_model,
        "api_key_configured": bool(key and not is_placeholder_key(key)) or bool(settings.openai_base_url),
    }


@app.get("/examples", include_in_schema=False)
def list_examples() -> list[dict]:
    """Примеры для кнопок на странице. Битый JSON отдаём как текст: на нём видно обработку ошибок."""
    examples = []
    for path in sorted((EXAMPLES_DIR / "products").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        try:
            title = json.loads(text).get("name") or path.stem
        except (json.JSONDecodeError, AttributeError):
            title = path.stem
        examples.append({"id": path.stem, "title": title, "text": text})

    for stem, title in EDGE_CASE_TITLES.items():
        path = EXAMPLES_DIR / "edge_cases" / f"{stem}.json"
        if path.exists():
            examples.append({"id": stem, "title": title, "text": path.read_text(encoding="utf-8")})
    return examples

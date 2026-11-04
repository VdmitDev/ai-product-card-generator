"""Работа с OpenAI: запрос, проверка ответа, повторные попытки и понятные ошибки."""

import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import openai
from openai import OpenAI
from pydantic import ValidationError

from app.card_builder import build_result, tidy_draft
from app.checks import Issue, check_draft, describe_input_gaps, issues_weight
from app.config import Settings, get_settings, is_placeholder_key
from app.prompts import SYSTEM_PROMPT, build_format_repair_prompt, build_repair_prompt, build_user_prompt
from app.schemas import CardDraft, GenerationResult, ProductInput, card_draft_json_schema

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Structured Outputs: API гарантирует JSON строго по схеме CardDraft
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "product_card_draft",
        "strict": True,
        "schema": card_draft_json_schema(),
    },
}

CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMServiceError(Exception):
    """Ошибка генерации с текстом, который можно показать пользователю как есть."""

    def __init__(self, message: str, *, status_code: int = 502, error_code: str = "llm_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class DraftFormatError(ValueError):
    """Ответ модели не разбирается как JSON по схеме."""


@dataclass
class _Attempt:
    number: int
    draft: CardDraft
    issues: list[Issue]


def parse_draft(content: str) -> CardDraft:
    # Некоторые OpenAI-совместимые модели заворачивают JSON в ```json ... ```, хотя их об этом не просили
    text = CODE_FENCE.sub("", content.strip())
    try:
        return CardDraft.model_validate_json(text)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'ответ'}: {error['msg']}"
            for error in exc.errors()[:3]
        )
        raise DraftFormatError(problems) from exc


def _create_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key and not settings.openai_base_url:
        raise LLMServiceError(
            "Не задан OPENAI_API_KEY. Скопируйте .env.example в .env, впишите ключ и перезапустите сервис.",
            status_code=500,
            error_code="config_error",
        )
    if settings.openai_api_key and is_placeholder_key(settings.openai_api_key):
        raise LLMServiceError(
            "В OPENAI_API_KEY осталась заглушка из .env.example. Впишите настоящий ключ и перезапустите сервис.",
            status_code=500,
            error_code="config_error",
        )
    return OpenAI(
        # Локальным OpenAI-совместимым серверам (Ollama, vLLM) ключ не нужен, но SDK требует строку
        api_key=settings.openai_api_key or "not-needed",
        # Адрес передаём явно: пустой OPENAI_BASE_URL= из .env SDK принял бы за адрес ""
        base_url=settings.openai_base_url or OPENAI_DEFAULT_BASE_URL,
        timeout=settings.request_timeout,
        max_retries=2,  # SDK сам повторяет запрос при обрыве связи, 429 и ошибках 5xx
    )


class ProductCardGenerator:
    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.settings = settings
        self.client = client or _create_client(settings)
        # Необязательные параметры. Если модель какой-то из них не принимает, убираем его на лету
        self._optional_params: dict[str, object] = {}
        if settings.reasoning_effort:
            self._optional_params["reasoning_effort"] = settings.reasoning_effort
        if settings.temperature is not None:
            self._optional_params["temperature"] = settings.temperature

    def generate(self, product: ProductInput) -> GenerationResult:
        started_at = time.monotonic()
        input_gaps = describe_input_gaps(product)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(product, input_gaps)},
        ]
        max_attempts = self.settings.max_repair_attempts + 1
        best: _Attempt | None = None
        prompt_tokens = completion_tokens = 0
        attempt = 0

        for attempt in range(1, max_attempts + 1):
            content, (used_prompt, used_completion) = self._complete(messages)
            prompt_tokens += used_prompt
            completion_tokens += used_completion
            messages.append({"role": "assistant", "content": content})

            try:
                draft = tidy_draft(parse_draft(content))
            except DraftFormatError as exc:
                logger.warning("Попытка %s: ответ не по схеме (%s)", attempt, exc)
                messages.append({"role": "user", "content": build_format_repair_prompt(str(exc))})
                continue

            issues = check_draft(draft, product)
            # При равном счёте берём более позднюю попытку: в ней модель уже учла замечания
            if best is None or issues_weight(issues) <= issues_weight(best.issues):
                best = _Attempt(attempt, draft, issues)
            if not issues:
                break
            logger.info("Попытка %s не прошла проверку: %s", attempt, " | ".join(i.problem for i in issues))
            messages.append({"role": "user", "content": build_repair_prompt(issues)})

        if best is None:
            raise LLMServiceError(
                "Модель несколько раз вернула ответ не по схеме. Повторите запрос или выберите другую модель.",
                error_code="llm_invalid_response",
            )

        logger.info(
            "Карточка «%s»: попыток %s, взята попытка %s, замечаний %s",
            product.name, attempt, best.number, len(best.issues),
        )
        return build_result(
            product=product,
            draft=best.draft,
            issues=best.issues,
            input_gaps=input_gaps,
            model_name=self.settings.openai_model,
            attempts=attempt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=time.monotonic() - started_at,
        )

    def _complete(self, messages: list[dict]) -> tuple[str, tuple[int, int]]:
        """Один запрос к API. Возвращает текст ответа и расход токенов (prompt, completion)."""
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                response_format=RESPONSE_FORMAT,
                **self._optional_params,
            )
        except openai.BadRequestError as exc:
            rejected = self._find_rejected_param(exc)
            if rejected is None:
                raise self._to_service_error(exc) from exc
            # Например, reasoning-модели не принимают temperature, а старые модели reasoning_effort
            logger.warning(
                "Модель %s не принимает параметр %s, повторяю запрос без него",
                self.settings.openai_model, rejected,
            )
            self._optional_params.pop(rejected, None)
            return self._complete(messages)
        except openai.OpenAIError as exc:
            raise self._to_service_error(exc) from exc

        usage = response.usage
        tokens = ((usage.prompt_tokens or 0), (usage.completion_tokens or 0)) if usage else (0, 0)

        if not response.choices:
            raise LLMServiceError("Модель вернула пустой ответ.", error_code="llm_invalid_response")
        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMServiceError(
                f"Модель отказалась отвечать: {choice.message.refusal}", error_code="llm_refusal"
            )
        if choice.finish_reason == "length":
            raise LLMServiceError(
                "Ответ модели оборвался на лимите токенов. Сократите описание товара или выберите другую модель.",
                error_code="llm_truncated",
            )
        if choice.finish_reason == "content_filter":
            raise LLMServiceError("Ответ модели заблокировал фильтр контента.", error_code="llm_content_filter")
        if not choice.message.content:
            raise LLMServiceError("Модель вернула пустой ответ.", error_code="llm_invalid_response")
        return choice.message.content, tokens

    def _find_rejected_param(self, exc: openai.BadRequestError) -> str | None:
        message = str(exc.message or "")
        for name in self._optional_params:
            if exc.param == name or name in message:
                return name
        return None

    def _to_service_error(self, exc: openai.OpenAIError) -> LLMServiceError:
        """Переводит исключения SDK в ошибки с понятным текстом и HTTP-кодом для нашего API."""
        model = self.settings.openai_model
        # APITimeoutError наследуется от APIConnectionError, поэтому проверяем его первым
        if isinstance(exc, openai.APITimeoutError):
            return LLMServiceError(
                "OpenAI не ответил вовремя. Повторите запрос чуть позже.", status_code=504, error_code="llm_timeout"
            )
        if isinstance(exc, openai.APIConnectionError):
            return LLMServiceError(
                "Нет соединения с OpenAI API. Проверьте интернет, прокси или OPENAI_BASE_URL.",
                status_code=503, error_code="llm_unavailable",
            )
        if isinstance(exc, openai.AuthenticationError):
            return LLMServiceError(
                "OpenAI отклонил API-ключ. Проверьте OPENAI_API_KEY в .env.",
                status_code=500, error_code="llm_auth_error",
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return LLMServiceError(
                f"Нет доступа к модели {model}: проверьте права проекта в OpenAI или регион.",
                status_code=500, error_code="llm_permission_denied",
            )
        if isinstance(exc, openai.NotFoundError):
            return LLMServiceError(
                f"Модель {model} не найдена или недоступна. Проверьте OPENAI_MODEL в .env.",
                status_code=500, error_code="llm_model_not_found",
            )
        if isinstance(exc, openai.RateLimitError):
            if exc.code == "insufficient_quota":
                return LLMServiceError(
                    "На аккаунте OpenAI закончились средства или квота. Пополните баланс и повторите.",
                    status_code=503, error_code="llm_quota_exceeded",
                )
            return LLMServiceError(
                "Превышен лимит запросов к OpenAI. Подождите минуту и повторите.",
                status_code=503, error_code="llm_rate_limited",
            )
        if isinstance(exc, openai.BadRequestError):
            return LLMServiceError(
                f"OpenAI отклонил запрос: {exc.message}", status_code=502, error_code="llm_bad_request"
            )
        if isinstance(exc, openai.APIStatusError):
            return LLMServiceError(
                f"Ошибка на стороне OpenAI (HTTP {exc.status_code}). Повторите запрос позже.",
                status_code=502, error_code="llm_upstream_error",
            )
        return LLMServiceError(f"Ошибка при обращении к OpenAI: {exc}", status_code=502, error_code="llm_error")


@lru_cache
def get_generator() -> ProductCardGenerator:
    return ProductCardGenerator(get_settings())


def generate_product_card(product_data: ProductInput | dict) -> GenerationResult:
    """Главная функция сервиса: данные о товаре на входе, черновик карточки на выходе.

    Принимает уже проверенный ProductInput или «сырой» словарь. Словарь проверяется
    здесь же, при ошибке будет pydantic.ValidationError.
    """
    product = product_data if isinstance(product_data, ProductInput) else ProductInput.model_validate(product_data)
    return get_generator().generate(product)

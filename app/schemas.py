"""Pydantic-модели: входные данные, ответ модели и ответ API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Так обычно помечают пустые ячейки в таблицах поставщиков и выгрузках из учётных систем.
# Среднее и длинное тире записаны кодами, чтобы их было видно в исходнике.
EMPTY_VALUE_MARKERS = {
    "", "-", "\u2013", "\u2014", "н/д", "нет данных", "данных нет", "n/a", "na", "null", "none",
}

EXAMPLE_PRODUCT = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": (
        "Прибор предназначен для измерения избыточного давления неагрессивных, "
        "некристаллизующихся жидкостей, газов и пара."
    ),
}


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in EMPTY_VALUE_MARKERS


def _spec_value_to_text(key: str, value: Any) -> str:
    """Приводит значение характеристики к строке. Пустое значение становится ""."""
    if value is None:
        return ""
    # bool проверяем раньше чисел: в Python True тоже int
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "" if is_empty_value(value) else value.strip()
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float)) and not isinstance(item, bool) for item in value
    ):
        return ", ".join(str(item).strip() for item in value if not is_empty_value(str(item)))
    raise ValueError(f"значение характеристики «{key}» должно быть строкой, числом или списком строк")


class ProductInput(BaseModel):
    """Данные о приборе, которые присылает контент-менеджер."""

    model_config = ConfigDict(
        # Лишние поля (цена, артикул) не ломают запрос, но попадают в предупреждения
        extra="allow",
        json_schema_extra={"examples": [EXAMPLE_PRODUCT]},
    )

    name: str = Field(max_length=300, description="Название товара")
    manufacturer: str | None = Field(default=None, max_length=200, description="Производитель")
    specs: dict[str, str] = Field(
        default_factory=dict, description="Характеристики в виде {название: значение}"
    )
    description: str | None = Field(
        default=None, max_length=20_000, description="Описание или фрагмент технического паспорта"
    )

    @field_validator("name", mode="before")
    @classmethod
    def check_name(cls, value: Any) -> Any:
        # Нестроковые значения пропускаем дальше: Pydantic сам скажет, что ждёт строку
        if isinstance(value, str):
            if is_empty_value(value):
                raise ValueError("название товара не может быть пустым")
            return value.strip()
        return value

    @field_validator("manufacturer", "description", mode="before")
    @classmethod
    def empty_text_to_none(cls, value: Any) -> Any:
        # Пустая строка, пробелы и «н/д» означают одно и то же: данных нет
        if isinstance(value, str):
            return None if is_empty_value(value) else value.strip()
        return value

    @field_validator("specs", mode="before")
    @classmethod
    def normalize_specs(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('specs должен быть объектом вида {"характеристика": "значение"}')
        normalized = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if key:  # характеристика без названия ни о чём не говорит
                normalized[key] = _spec_value_to_text(key, raw_value)
        return normalized

    @model_validator(mode="after")
    def check_enough_data(self) -> "ProductInput":
        # Без характеристик и описания модели не на что опереться: любой текст был бы выдумкой
        if not self.filled_specs and not self.description:
            raise ValueError(
                "недостаточно данных для карточки: заполните хотя бы specs или description"
            )
        return self

    @property
    def filled_specs(self) -> dict[str, str]:
        return {key: value for key, value in self.specs.items() if value}

    @property
    def empty_spec_names(self) -> list[str]:
        return [key for key, value in self.specs.items() if not value]

    @property
    def ignored_fields(self) -> list[str]:
        return sorted(self.model_extra or {})


# --- Ответ модели -----------------------------------------------------------
# Эти модели превращаются в JSON Schema для Structured Outputs (strict).
# Strict-режиму нужны две вещи: все поля обязательные (значений по умолчанию нет)
# и additionalProperties: false, его даёт extra="forbid".


class ExtractedSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str
    quote: str


class MissingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    reason: str


class CommercialBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_audience: list[str] = Field(description="Кому подходит прибор")
    tasks_solved: list[str] = Field(description="Какие задачи решает")


# Черновик от модели. Порядок полей важен, почему именно такой: см. prompts.py.
# Комментарий, а не docstring: docstring Pydantic вставил бы в схему, и его прочитала бы модель.
class CardDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specs_from_description: list[ExtractedSpec]
    missing_specs: list[MissingSpec]
    clarifications: list[str]
    search_queries: list[str]
    seo_title: str
    seo_description: str
    commercial_block: CommercialBlock


def card_draft_json_schema() -> dict:
    return CardDraft.model_json_schema()


# --- Ответ API --------------------------------------------------------------


class SpecRow(BaseModel):
    name: str
    value: str
    source: Literal["input", "description", "missing"] = Field(
        description="input: из specs; description: из текста описания; missing: данных нет"
    )
    note: str | None = Field(
        default=None,
        description="Для description: цитата из описания. Для missing: зачем нужна характеристика",
    )


class ProductCard(BaseModel):
    seo_title: str
    seo_description: str
    key_specs: list[SpecRow]
    search_queries: list[str]
    commercial_block: CommercialBlock


class DataQuality(BaseModel):
    needs_clarification: bool = Field(description="Есть пробелы или замечания, карточку нужно проверить")
    missing_fields: list[str] = Field(description="Незаполненные поля входного JSON")
    missing_specs: list[str] = Field(description="Характеристики, у которых в карточке стоит «данных нет»")
    clarifications: list[str] = Field(description="Противоречия и неясности во входных данных")


class GenerationMeta(BaseModel):
    model: str
    attempts: int = Field(description="Сколько запросов к модели понадобилось")
    seo_title_length: int
    seo_description_length: int
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    generated_at: datetime


class GenerationResult(BaseModel):
    product_name: str
    card: ProductCard
    data_quality: DataQuality
    warnings: list[str] = Field(description="Что не прошло автоматическую проверку")
    markdown: str
    meta: GenerationMeta


# --- Человекочитаемые ошибки валидации ---------------------------------------

_ERROR_TEXTS = {
    "missing": "обязательное поле отсутствует",
    "string_type": "ожидается строка",
    "dict_type": "ожидается JSON-объект",
    "model_attributes_type": "ожидается JSON-объект с полями name, manufacturer, specs, description",
    "model_type": "ожидается JSON-объект с полями name, manufacturer, specs, description",
}


def _position_in_text(text: str, position: int) -> str:
    line = text.count("\n", 0, position) + 1
    column = position - (text.rfind("\n", 0, position) + 1) + 1
    return f"строка {line}, позиция {column}"


def format_validation_errors(errors: list[dict], raw_body: Any = None) -> list[str]:
    """Переводит ошибки Pydantic и FastAPI в понятные контент-менеджеру фразы."""
    messages = []
    for error in errors:
        error_type = error.get("type", "")
        context = error.get("ctx") or {}
        location = [str(part) for part in error.get("loc", ()) if part != "body"]

        if error_type == "json_invalid":
            # Для битого JSON в loc лежит номер символа, а не имя поля
            where = ""
            if location and location[0].isdigit() and isinstance(raw_body, str):
                where = f" ({_position_in_text(raw_body, int(location[0]))})"
            messages.append(f"ошибка синтаксиса JSON: {context.get('error', 'не удалось разобрать')}{where}")
            continue

        if error_type == "value_error":
            text = str(context.get("error") or error.get("msg", ""))
        elif error_type == "string_too_long":
            text = f"слишком длинное значение, максимум {context.get('max_length')} символов"
        else:
            text = _ERROR_TEXTS.get(error_type, error.get("msg", ""))

        field = ".".join(location) if location else "запрос"
        messages.append(f"{field}: {text}")
    return messages

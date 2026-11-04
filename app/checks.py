"""Автоматические проверки входных данных и ответа модели.

Промпт снижает риск выдумок, но не гарантирует их отсутствие. Поэтому всё, что
можно проверить кодом (длины, числа, цитаты), проверяем кодом, а не верим модели.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.rules import (
    DESCRIPTION_MAX_LENGTH,
    DESCRIPTION_MIN_LENGTH,
    DESCRIPTION_TARGET_LENGTH,
    NO_DATA,
    SEARCH_QUERIES_MAX,
    SEARCH_QUERIES_MIN,
    TITLE_MAX_LENGTH,
)
from app.schemas import CardDraft, ExtractedSpec, ProductInput

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
# «B2B» содержит цифру, но к характеристикам прибора отношения не имеет
NON_SPEC_TOKENS = re.compile(r"\bB2[BCG]\b", re.IGNORECASE)
DASHES = re.compile(r"[\u2010-\u2015\u2212]")
QUOTE_MARKS = re.compile(r"[«»„“”\"']")

FIELD_LABELS = {
    "seo_title": "SEO-заголовок",
    "seo_description": "SEO-описание",
    "search_queries": "Поисковые запросы",
    "commercial_block": "Блок для менеджера",
    "specs_from_description": "Характеристики из описания",
}

# Подозрение на выдумку весит больше, чем промах по длине: при выборе лучшей
# из нескольких попыток честный текст на 780 символов лучше текста с «IP65».
WEIGHT_MINOR = 1
WEIGHT_FACTUAL = 3


@dataclass(frozen=True)
class Issue:
    field: str
    problem: str  # что не так: уходит в предупреждения для контент-менеджера
    fix_hint: str  # что исправить: уходит в повторный запрос к модели
    weight: int = WEIGHT_MINOR


def issues_weight(issues: list[Issue]) -> int:
    return sum(issue.weight for issue in issues)


# --- Входные данные ------------------------------------------------------------


def describe_input_gaps(product: ProductInput) -> list[str]:
    """Пробелы в полях входного JSON. Пустые характеристики сюда не входят:
    они попадают в таблицу со значением «данных нет»."""
    gaps = []
    if not product.manufacturer:
        gaps.append("Не указан производитель (manufacturer).")
    if not product.description:
        gaps.append("Нет текстового описания (description): карточка строится только по характеристикам.")
    if not product.filled_specs:
        gaps.append("Не заполнены характеристики (specs): значения берутся только из описания.")
    return gaps


# --- Нормализация ------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Приводит текст к виду для сравнения: регистр, ё, тире, кавычки и пробелы не важны."""
    text = text.lower().replace("ё", "е")
    text = DASHES.sub("-", text)
    text = QUOTE_MARKS.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:")


def _canonical_number(token: str) -> str:
    try:
        return format(Decimal(token.replace(",", ".")).normalize(), "f")
    except InvalidOperation:
        return token


def extract_numbers(text: str) -> dict[str, str]:
    """Числа из текста: канонический вид -> как записано. «1,6» и «1.6» считаются одним числом."""
    cleaned = NON_SPEC_TOKENS.sub(" ", text)
    return {_canonical_number(token): token for token in NUMBER_PATTERN.findall(cleaned)}


def _source_text(product: ProductInput) -> str:
    parts = [product.name, product.manufacturer or "", product.description or ""]
    for key, value in product.specs.items():
        parts += [key, value]
    return "\n".join(parts)


# --- Ответ модели --------------------------------------------------------------------


def split_extracted_specs(
    draft: CardDraft, product: ProductInput
) -> tuple[list[ExtractedSpec], list[ExtractedSpec]]:
    """Делит характеристики из описания на подтверждённые цитатой и неподтверждённые."""
    description = normalize_text(product.description or "")
    verified, rejected = [], []
    for spec in draft.specs_from_description:
        quote = normalize_text(spec.quote)
        quote_found = bool(description) and len(quote) >= 3 and quote in description
        # Числа в значении должны быть в самой цитате, иначе модель могла их «подправить»
        numbers_match = extract_numbers(spec.value).keys() <= extract_numbers(spec.quote).keys()
        (verified if quote_found and numbers_match else rejected).append(spec)
    return verified, rejected


def find_unknown_numbers(draft: CardDraft, product: ProductInput) -> dict[str, list[str]]:
    """Числа в сгенерированных текстах, которых нет во входных данных, по полям."""
    known_numbers = extract_numbers(_source_text(product)).keys()
    generated = [("seo_title", draft.seo_title), ("seo_description", draft.seo_description)]
    generated += [("search_queries", query) for query in draft.search_queries]
    block = draft.commercial_block
    generated += [("commercial_block", item) for item in block.target_audience + block.tasks_solved]
    generated += [("specs_from_description", spec.value) for spec in draft.specs_from_description]

    unknown: dict[str, list[str]] = {}
    for field, text in generated:
        for canonical, original in extract_numbers(text).items():
            if canonical not in known_numbers and original not in unknown.get(field, []):
                unknown.setdefault(field, []).append(original)
    return unknown


def check_draft(draft: CardDraft, product: ProductInput) -> list[Issue]:
    return [
        *_check_title(draft),
        *_check_description(draft),
        *_check_search_queries(draft),
        *_check_commercial_block(draft),
        *_check_extracted_specs(draft, product),
        *_check_unknown_numbers(draft, product),
    ]


def _check_title(draft: CardDraft) -> list[Issue]:
    length = len(draft.seo_title)
    if length == 0:
        return [Issue("seo_title", "SEO-заголовок пустой.", "Заполни seo_title по формуле из инструкции.", WEIGHT_FACTUAL)]
    if length > TITLE_MAX_LENGTH:
        return [Issue(
            "seo_title",
            f"SEO-заголовок длиннее {TITLE_MAX_LENGTH} символов ({length}).",
            f"seo_title: {length} символов, допустимо не больше {TITLE_MAX_LENGTH}. Сократи минимум на "
            f"{length - TITLE_MAX_LENGTH}: убери производителя или второстепенный параметр, тип прибора и модель оставь.",
        )]
    return []


def _check_description(draft: CardDraft) -> list[Issue]:
    issues = []
    length = len(draft.seo_description)
    if length < DESCRIPTION_MIN_LENGTH:
        issues.append(Issue(
            "seo_description",
            f"SEO-описание короче {DESCRIPTION_MIN_LENGTH} символов ({length}). Если фактов во входных "
            f"данных больше нет, допишите текст вручную после уточнения характеристик.",
            f"seo_description: {length} символов, нужно от {DESCRIPTION_MIN_LENGTH} до {DESCRIPTION_MAX_LENGTH}. "
            f"Допиши примерно {DESCRIPTION_TARGET_LENGTH - length} символов: подробнее раскрой назначение "
            f"и область применения из входных данных. Новых характеристик не добавляй.",
        ))
    elif length > DESCRIPTION_MAX_LENGTH:
        issues.append(Issue(
            "seo_description",
            f"SEO-описание длиннее {DESCRIPTION_MAX_LENGTH} символов ({length}).",
            f"seo_description: {length} символов, допустимо не больше {DESCRIPTION_MAX_LENGTH}. "
            f"Сократи примерно на {length - DESCRIPTION_TARGET_LENGTH} символов, сохранив все характеристики.",
        ))
    if NO_DATA in draft.seo_description.lower():
        issues.append(Issue(
            "seo_description",
            f"В SEO-описании есть пометка «{NO_DATA}», а это публичный текст.",
            f"Убери из seo_description упоминания отсутствующих характеристик и пометку «{NO_DATA}».",
        ))
    return issues


def _check_search_queries(draft: CardDraft) -> list[Issue]:
    count = len(draft.search_queries)
    if SEARCH_QUERIES_MIN <= count <= SEARCH_QUERIES_MAX:
        return []
    return [Issue(
        "search_queries",
        f"Поисковых запросов {count}, а нужно от {SEARCH_QUERIES_MIN} до {SEARCH_QUERIES_MAX}.",
        f"search_queries: сейчас {count} разных запросов, верни от {SEARCH_QUERIES_MIN} до {SEARCH_QUERIES_MAX}.",
    )]


def _check_commercial_block(draft: CardDraft) -> list[Issue]:
    issues = []
    if not draft.commercial_block.target_audience:
        issues.append(Issue(
            "commercial_block",
            "В блоке для менеджера не заполнено, кому подходит прибор.",
            "commercial_block.target_audience пустой: добавь 2-4 пункта.",
        ))
    if not draft.commercial_block.tasks_solved:
        issues.append(Issue(
            "commercial_block",
            "В блоке для менеджера не заполнено, какие задачи решает прибор.",
            "commercial_block.tasks_solved пустой: добавь 2-4 пункта.",
        ))
    return issues


def _check_extracted_specs(draft: CardDraft, product: ProductInput) -> list[Issue]:
    _, rejected = split_extracted_specs(draft, product)
    return [
        Issue(
            "specs_from_description",
            f"«{spec.name}: {spec.value}» не подтверждается дословной цитатой из описания, "
            f"в карточку не включено. Проверьте вручную.",
            f"specs_from_description: для «{spec.name}» цитата не найдена в description или не содержит "
            f"значения. Скопируй фрагмент description символ в символ или убери эту характеристику.",
            WEIGHT_FACTUAL,
        )
        for spec in rejected
    ]


def _check_unknown_numbers(draft: CardDraft, product: ProductInput) -> list[Issue]:
    issues = []
    for field, numbers in find_unknown_numbers(draft, product).items():
        listed = ", ".join(numbers)
        issues.append(Issue(
            field,
            f"{FIELD_LABELS[field]}: числа, которых нет во входных данных ({listed}). "
            f"Проверьте, не выдумана ли характеристика.",
            f"{field}: есть числа, которых нет во входных данных: {listed}. "
            f"Убери их или перепиши фразу без цифр.",
            WEIGHT_FACTUAL,
        ))
    return issues

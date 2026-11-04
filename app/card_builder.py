"""Сборка итоговой карточки из черновика модели и входных данных."""

import re
from datetime import datetime, timezone

from app.checks import Issue, split_extracted_specs
from app.markdown_renderer import render_markdown
from app.rules import MANUFACTURER_LABEL, MISSING_SPECS_MAX, NO_DATA
from app.schemas import (
    CardDraft,
    CommercialBlock,
    DataQuality,
    GenerationMeta,
    GenerationResult,
    ProductCard,
    ProductInput,
    SpecRow,
)


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_list(items: list[str]) -> list[str]:
    """Убирает пустые пункты и повторы, сохраняя порядок."""
    result, seen = [], set()
    for item in items:
        item = _collapse_spaces(item)
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def _clean_description(text: str) -> str:
    paragraphs = [_collapse_spaces(part) for part in re.split(r"\n\s*\n", text.strip())]
    return "\n\n".join(part for part in paragraphs if part)


def tidy_draft(draft: CardDraft) -> CardDraft:
    """Косметика, которую надёжнее сделать кодом, чем просить у модели: пробелы,
    точка в конце заголовка, повторы и регистр поисковых запросов."""
    return draft.model_copy(update={
        "seo_title": _collapse_spaces(draft.seo_title).rstrip("."),
        "seo_description": _clean_description(draft.seo_description),
        "search_queries": _clean_list([query.lower() for query in draft.search_queries]),
        "clarifications": _clean_list(draft.clarifications),
        "commercial_block": CommercialBlock(
            target_audience=_clean_list(draft.commercial_block.target_audience),
            tasks_solved=_clean_list(draft.commercial_block.tasks_solved),
        ),
    })


def _spec_key(name: str) -> str:
    """Ключ для поиска дублей: «Класс точности» и «класс точности:» считаем одним и тем же."""
    return re.sub(r"[\W_]+", " ", name.lower().replace("ё", "е")).strip()


def _capitalize(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:]


def _lower_first(text: str) -> str:
    """Пояснение идёт после двоеточия: «Нужна для...» -> «нужна для...».
    Аббревиатуры («IP», «ГОСТ») не трогаем: у них вторая буква тоже заглавная."""
    text = text.strip()
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def build_spec_rows(draft: CardDraft, product: ProductInput) -> list[SpecRow]:
    rows = [
        SpecRow(name=MANUFACTURER_LABEL, value=product.manufacturer, source="input")
        if product.manufacturer
        else SpecRow(name=MANUFACTURER_LABEL, value=NO_DATA, source="missing")
    ]

    # Значения из specs переносим как есть, без участия модели: так их нельзя исказить
    for name, value in product.specs.items():
        if value:
            rows.append(SpecRow(name=_capitalize(name), value=value, source="input"))
        else:
            rows.append(SpecRow(
                name=_capitalize(name), value=NO_DATA, source="missing",
                note="значение не заполнено во входных данных",
            ))

    seen = {_spec_key(row.name) for row in rows}
    verified_specs, _ = split_extracted_specs(draft, product)
    for spec in verified_specs:
        key = _spec_key(spec.name)
        if key and key not in seen:
            seen.add(key)
            rows.append(SpecRow(
                name=_capitalize(spec.name), value=spec.value.strip(),
                source="description", note=spec.quote.strip(),
            ))

    for missing in draft.missing_specs[:MISSING_SPECS_MAX]:
        key = _spec_key(missing.name)
        if key and key not in seen:
            seen.add(key)
            rows.append(SpecRow(
                name=_capitalize(missing.name), value=NO_DATA,
                source="missing", note=_lower_first(missing.reason) or None,
            ))
    return rows


def build_result(
    *,
    product: ProductInput,
    draft: CardDraft,
    issues: list[Issue],
    input_gaps: list[str],
    model_name: str,
    attempts: int,
    prompt_tokens: int,
    completion_tokens: int,
    duration_seconds: float,
) -> GenerationResult:
    key_specs = build_spec_rows(draft, product)
    card = ProductCard(
        seo_title=draft.seo_title,
        seo_description=draft.seo_description,
        key_specs=key_specs,
        search_queries=draft.search_queries,
        commercial_block=draft.commercial_block,
    )

    warnings = [issue.problem for issue in issues]
    if product.ignored_fields:
        warnings.append(
            f"Поля {', '.join(product.ignored_fields)} не входят в формат и при генерации не использовались."
        )

    # Производитель уже есть в missing_fields, второй раз его не показываем
    missing_specs = [
        row.name for row in key_specs if row.source == "missing" and row.name != MANUFACTURER_LABEL
    ]
    data_quality = DataQuality(
        needs_clarification=bool(input_gaps or missing_specs or draft.clarifications or warnings),
        missing_fields=input_gaps,
        missing_specs=missing_specs,
        clarifications=draft.clarifications,
    )

    meta = GenerationMeta(
        model=model_name,
        attempts=attempts,
        seo_title_length=len(card.seo_title),
        seo_description_length=len(card.seo_description),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_seconds=round(duration_seconds, 1),
        generated_at=datetime.now(timezone.utc).replace(microsecond=0),
    )

    return GenerationResult(
        product_name=product.name,
        card=card,
        data_quality=data_quality,
        warnings=warnings,
        markdown=render_markdown(product.name, card, data_quality, warnings, meta),
        meta=meta,
    )

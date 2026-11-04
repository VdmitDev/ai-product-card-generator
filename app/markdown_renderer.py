"""Карточка в Markdown: удобно вставить в CMS, задачу или переписку с коллегами."""

from app.rules import DESCRIPTION_MAX_LENGTH, DESCRIPTION_MIN_LENGTH, MANUFACTURER_LABEL, TITLE_MAX_LENGTH
from app.schemas import DataQuality, GenerationMeta, ProductCard

CHARACTER_FORMS = ("символ", "символа", "символов")


def plural_ru(number: int, forms: tuple[str, str, str]) -> str:
    """plural_ru(21, CHARACTER_FORMS) -> «символ», 23 -> «символа», 25 -> «символов»."""
    if number % 10 == 1 and number % 100 != 11:
        return forms[0]
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return forms[1]
    return forms[2]


def _cell(text: str) -> str:
    # Вертикальная черта внутри значения сломала бы таблицу
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _review_section(card: ProductCard, data_quality: DataQuality, warnings: list[str]) -> list[str]:
    missing_rows = [
        row for row in card.key_specs if row.source == "missing" and row.name != MANUFACTURER_LABEL
    ]
    description_rows = [row for row in card.key_specs if row.source == "description"]

    lines = ["## Что проверить перед публикацией", ""]
    if not (data_quality.missing_fields or missing_rows or description_rows
            or data_quality.clarifications or warnings):
        return lines + ["Замечаний нет.", ""]

    if data_quality.missing_fields:
        lines += ["**Пробелы во входных данных:**", "", *_bullets(data_quality.missing_fields), ""]
    if missing_rows:
        lines += ["**Нет данных, запросите у поставщика:**", ""]
        lines += [f"- {row.name}" + (f": {row.note}" if row.note else "") for row in missing_rows]
        lines.append("")
    if description_rows:
        lines += ["**Взято из текста описания, сверьте формулировку:**", ""]
        lines += [f"- {row.name}: {row.value} (в описании: «{row.note}»)" for row in description_rows]
        lines.append("")
    if data_quality.clarifications:
        lines += ["**Противоречия и неясности:**", "", *_bullets(data_quality.clarifications), ""]
    if warnings:
        lines += ["**Автоматическая проверка:**", "", *_bullets(warnings), ""]
    return lines


def render_markdown(
    product_name: str,
    card: ProductCard,
    data_quality: DataQuality,
    warnings: list[str],
    meta: GenerationMeta,
) -> str:
    title_length = len(card.seo_title)
    description_length = len(card.seo_description)

    lines = [f"# {product_name}", ""]

    lines += [
        "## SEO-заголовок", "", card.seo_title, "",
        f"_{title_length} из {TITLE_MAX_LENGTH} символов_", "",
    ]
    lines += [
        "## SEO-описание", "", card.seo_description, "",
        f"_{description_length} {plural_ru(description_length, CHARACTER_FORMS)}, "
        f"норма {DESCRIPTION_MIN_LENGTH}-{DESCRIPTION_MAX_LENGTH}_", "",
    ]

    lines += ["## Ключевые характеристики", "", "| Характеристика | Значение |", "|---|---|"]
    lines += [f"| {_cell(row.name)} | {_cell(row.value)} |" for row in card.key_specs]
    lines.append("")

    lines += ["## Поисковые запросы", ""]
    lines += [f"{number}. {query}" for number, query in enumerate(card.search_queries, start=1)]
    lines.append("")

    lines += [
        "## Для менеджера по продажам", "",
        "**Кому подходит:**", "", *_bullets(card.commercial_block.target_audience), "",
        "**Какие задачи решает:**", "", *_bullets(card.commercial_block.tasks_solved), "",
    ]

    lines += _review_section(card, data_quality, warnings)

    lines += [
        "---", "",
        f"_Черновик подготовлен автоматически: {meta.model}, {meta.generated_at:%d.%m.%Y}. "
        f"Технические значения взяты только из входных данных._",
    ]
    return "\n".join(lines) + "\n"

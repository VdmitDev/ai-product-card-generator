"""Пакетная генерация карточек из JSON-файлов.

Примеры:
    python cli.py examples/products/*.json
    python cli.py my_products.json --out results

В файле может лежать один товар (объект) или несколько (массив объектов).
Для каждого товара сохраняются <имя>.json (полный ответ сервиса) и <имя>.md (карточка).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from app.llm_service import LLMServiceError, generate_product_card
from app.schemas import ProductInput, format_validation_errors

# После этих ошибок остальные файлы упадут так же, продолжать нет смысла
FATAL_ERROR_CODES = {
    "config_error", "llm_auth_error", "llm_quota_exceeded", "llm_model_not_found", "llm_permission_denied",
}


class InputFileError(Exception):
    pass


def read_items(path: Path) -> list[tuple[str, object]]:
    """Читает файл и возвращает пары (имя для результата, сырые данные товара)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InputFileError("файл не найден") from None
    except json.JSONDecodeError as exc:
        raise InputFileError(f"некорректный JSON: {exc.msg} (строка {exc.lineno}, позиция {exc.colno})") from None

    items = data if isinstance(data, list) else [data]
    if len(items) == 1:
        return [(path.stem, items[0])]
    return [(f"{path.stem}_{number}", item) for number, item in enumerate(items, start=1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Генерация черновиков карточек товаров КИП")
    parser.add_argument("files", nargs="+", type=Path, help="JSON-файлы с данными товаров")
    parser.add_argument("-o", "--out", type=Path, default=Path("examples/output"), help="папка для результатов")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    # Строки «HTTP Request: POST ...» в консоли только мешают
    for noisy_logger in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0

    for path in args.files:
        try:
            items = read_items(path)
        except InputFileError as exc:
            print(f"[ошибка] {path}: {exc}", file=sys.stderr)
            failed += 1
            continue

        for output_name, raw_item in items:
            try:
                product = ProductInput.model_validate(raw_item)
            except ValidationError as exc:
                print(f"[ошибка] {output_name}: {'; '.join(format_validation_errors(exc.errors()))}", file=sys.stderr)
                failed += 1
                continue

            print(f"[...] {output_name}: {product.name}", flush=True)
            try:
                result = generate_product_card(product)
            except LLMServiceError as exc:
                print(f"[ошибка] {output_name}: {exc.message}", file=sys.stderr)
                failed += 1
                if exc.error_code in FATAL_ERROR_CODES:
                    return 1
                continue

            json_path = args.out / f"{output_name}.json"
            json_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
            (args.out / f"{output_name}.md").write_text(result.markdown, encoding="utf-8")

            meta = result.meta
            print(
                f"[ok] {output_name}: заголовок {meta.seo_title_length}/70, описание {meta.seo_description_length} симв., "
                f"запросов к модели {meta.attempts}, замечаний {len(result.warnings)} -> {json_path.with_suffix('.md')}"
            )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

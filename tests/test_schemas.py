import pytest
from pydantic import ValidationError

from app.schemas import ProductInput, format_validation_errors
from factories import MANOMETER, make_product


def test_input_is_normalized():
    product = ProductInput.model_validate({
        "name": "  Манометр DM 02 ",
        "manufacturer": "н/д",
        "specs": {
            "класс точности": 1.5,
            "с подсветкой": False,
            "исполнения": ["радиальное", "осевое"],
            "степень защиты": " - ",
            "  ": "характеристика без названия",
        },
        "description": "Прибор для измерения давления.",
    })

    assert product.name == "Манометр DM 02"
    assert product.manufacturer is None
    assert product.specs == {
        "класс точности": "1.5",
        "с подсветкой": "нет",
        "исполнения": "радиальное, осевое",
        "степень защиты": "",
    }
    assert product.empty_spec_names == ["степень защиты"]
    assert "степень защиты" not in product.filled_specs


@pytest.mark.parametrize("name", ["", "   ", "н/д", "\u2014"])
def test_empty_name_is_rejected(name):
    with pytest.raises(ValidationError, match="название товара не может быть пустым"):
        make_product(name=name)


def test_name_is_required():
    data = {key: value for key, value in MANOMETER.items() if key != "name"}
    with pytest.raises(ValidationError) as error:
        ProductInput.model_validate(data)
    assert format_validation_errors(error.value.errors()) == ["name: обязательное поле отсутствует"]


def test_specs_must_be_an_object():
    with pytest.raises(ValidationError, match="specs должен быть объектом"):
        make_product(specs=["0–1,6 МПа", "1,5"])


def test_nested_spec_value_is_rejected():
    with pytest.raises(ValidationError, match="значение характеристики «питание»"):
        make_product(specs={"питание": {"напряжение": "24 В"}})


def test_specs_or_description_required():
    with pytest.raises(ValidationError, match="недостаточно данных"):
        make_product(specs={"класс точности": ""}, description="  ")


def test_card_can_be_built_from_description_only():
    product = make_product(specs={})
    assert product.filled_specs == {}
    assert product.description


def test_extra_fields_are_kept_for_warnings():
    product = make_product(price="1 250 руб.", sku="DM02-100")
    assert product.ignored_fields == ["price", "sku"]


def test_validation_errors_are_human_readable():
    with pytest.raises(ValidationError) as error:
        ProductInput.model_validate({"name": 123, "specs": {"класс": "1,5"}, "description": "x" * 20_001})

    messages = format_validation_errors(error.value.errors())
    assert "name: ожидается строка" in messages
    assert "description: слишком длинное значение, максимум 20000 символов" in messages


def test_broken_json_position_is_reported():
    body = '{\n  "name": "Манометр",\n  "specs": {"класс": "1,5",}\n}'
    position = body.index("}")
    errors = [{"type": "json_invalid", "loc": ("body", position), "ctx": {"error": "Expecting property name"}}]

    assert format_validation_errors(errors, raw_body=body) == [
        "ошибка синтаксиса JSON: Expecting property name (строка 3, позиция 28)"
    ]

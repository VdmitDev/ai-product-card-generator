import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.llm_service import LLMServiceError
from factories import MANOMETER, completion_from_payload, make_draft_payload, make_generator

client = TestClient(main_module.app)


@pytest.fixture
def fake_generator(monkeypatch):
    generator, fake_client = make_generator(*[completion_from_payload(make_draft_payload())] * 3)
    monkeypatch.setattr(main_module, "get_generator", lambda: generator)
    return fake_client


def test_index_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Сгенерировать" in response.text


def test_generate_returns_card_as_json(fake_generator):
    response = client.post("/generate", json=MANOMETER)

    assert response.status_code == 200
    body = response.json()
    assert body["card"]["seo_title"] == make_draft_payload()["seo_title"]
    assert body["card"]["key_specs"][0] == {"name": "Производитель", "value": "Манотомь", "source": "input", "note": None}
    assert body["markdown"].startswith("# Манометр показывающий DM 02")


def test_generate_can_return_markdown(fake_generator):
    response = client.post("/generate?format=markdown", json=MANOMETER)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "## SEO-описание" in response.text


def test_broken_json_gets_clear_message(fake_generator):
    response = client.post(
        "/generate",
        content='{"name": "Манометр", "specs": {"класс": "1,5",}}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_json"
    assert "строка 1" in body["details"][0]
    assert fake_generator.calls == []


def test_validation_errors_are_listed(fake_generator):
    response = client.post("/generate", json={"name": " ", "specs": "диапазон 0–1,6 МПа"})

    assert response.status_code == 422
    details = response.json()["details"]
    assert "name: название товара не может быть пустым" in details
    assert any(detail.startswith("specs:") for detail in details)


def test_validation_errors_are_shown_even_without_api_key(monkeypatch):
    def broken_generator():
        raise LLMServiceError("Не задан OPENAI_API_KEY.", status_code=500, error_code="config_error")

    monkeypatch.setattr(main_module, "get_generator", broken_generator)

    assert client.post("/generate", json={"name": ""}).status_code == 422
    response = client.post("/generate", json=MANOMETER)
    assert response.status_code == 500
    assert response.json() == {"error": "config_error", "message": "Не задан OPENAI_API_KEY."}


def test_examples_include_products_and_edge_cases():
    examples = {item["id"]: item for item in client.get("/examples").json()}

    assert examples["01_manometer_dm02"]["title"] == "Манометр показывающий DM 02"
    assert examples["invalid_json"]["title"] == "Битый JSON"
    with pytest.raises(json.JSONDecodeError):
        json.loads(examples["invalid_json"]["text"])


def test_health_reports_model():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]

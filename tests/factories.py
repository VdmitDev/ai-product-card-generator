"""Заготовки для тестов: товар, ответ модели и поддельный клиент OpenAI."""

import copy
import json
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from app.config import Settings
from app.llm_service import ProductCardGenerator
from app.schemas import CardDraft, ProductInput

try:
    import httpx2 as http_client  # openai>=3 работает поверх httpx2
except ImportError:  # pragma: no cover
    import httpx as http_client

MANOMETER = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": (
        "Прибор предназначен для измерения избыточного давления неагрессивных, некристаллизующихся "
        "жидкостей, газов и пара. Применяется в системах водоснабжения и отопления, в котельных "
        "и на технологическом оборудовании промышленных предприятий."
    ),
}

# 954 символа, только факты из MANOMETER
GOOD_DESCRIPTION = (
    "Манометр показывающий DM 02 производства Манотомь предназначен для измерения избыточного давления "
    "неагрессивных, некристаллизующихся жидкостей, газов и пара. Прибор применяется в системах водоснабжения "
    "и отопления, в котельных и на технологическом оборудовании промышленных предприятий.\n\n"
    "Диапазон измерений составляет 0–1,6 МПа, класс точности 1,5. Присоединение к процессу выполнено "
    "резьбой G1/2. Корпус диаметром 100 мм, поэтому показания удобно считывать при обходе оборудования "
    "и при установке прибора на щит.\n\n"
    "Манометр DM 02 подойдёт службам эксплуатации зданий, котельным, монтажным и сервисным организациям, "
    "которым нужен прибор для визуального контроля давления. Его устанавливают на трубопроводы, насосные "
    "агрегаты и теплообменники, где важно видеть текущее давление в системе и вовремя замечать отклонения "
    "от рабочего режима. Перед заказом сверьте диапазон и присоединение с проектом: это основные параметры "
    "при подборе показывающего манометра."
)


def make_product(**overrides) -> ProductInput:
    data = copy.deepcopy(MANOMETER)
    data.update(overrides)
    return ProductInput.model_validate(data)


def make_draft_payload(**overrides) -> dict:
    payload = {
        "specs_from_description": [
            {
                "name": "Измеряемая среда",
                "value": "неагрессивные, некристаллизующиеся жидкости, газы и пар",
                "quote": "неагрессивных, некристаллизующихся жидкостей, газов и пара",
            }
        ],
        "missing_specs": [
            {"name": "Степень защиты", "reason": "Нужна, чтобы понять, можно ли ставить прибор на улице."},
            {"name": "Класс точности", "reason": "Дубль характеристики из specs, в карточку попасть не должен."},
        ],
        "clarifications": [],
        "search_queries": [
            "манометр dm 02",
            "манометр dm 02 0-1,6 мпа",
            "манометр манотомь 100 мм",
            "купить манометр dm 02",
        ],
        "seo_title": "Манометр показывающий DM 02 Манотомь, 0–1,6 МПа, кл. 1,5",
        "seo_description": GOOD_DESCRIPTION,
        "commercial_block": {
            "target_audience": ["Службы эксплуатации котельных", "Монтажные и сервисные организации"],
            "tasks_solved": ["Визуальный контроль давления в трубопроводах и на оборудовании"],
        },
    }
    payload.update(overrides)
    return payload


def make_draft(**overrides) -> CardDraft:
    return CardDraft.model_validate(make_draft_payload(**overrides))


def make_completion(content: str | None, *, finish_reason: str = "stop", refusal: str | None = None) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content, "refusal": refusal},
        }],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 400, "total_tokens": 1400},
    })


def completion_from_payload(payload: dict) -> ChatCompletion:
    return make_completion(json.dumps(payload, ensure_ascii=False))


def make_api_error(error_class, status_code: int, *, code=None, param=None, message="error"):
    request = http_client.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = http_client.Response(status_code, request=request)
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return error_class(message, response=response, body=body)


def make_request():
    return http_client.Request("POST", "https://api.openai.com/v1/chat/completions")


class FakeOpenAI:
    """Отдаёт заранее заготовленные ответы по очереди и запоминает запросы."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Копия обязательна: сервис дописывает в список messages после запроса
        self.calls.append(copy.deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "openai_base_url": None,
        "reasoning_effort": "low",
        "temperature": None,
        "request_timeout": 5.0,
        "max_repair_attempts": 2,
    }
    values.update(overrides)
    return Settings(**values)


def make_generator(*responses, **settings_overrides) -> tuple[ProductCardGenerator, FakeOpenAI]:
    client = FakeOpenAI(*responses)
    return ProductCardGenerator(make_settings(**settings_overrides), client=client), client

from app.checks import check_draft, describe_input_gaps, extract_numbers, find_unknown_numbers, split_extracted_specs
from factories import GOOD_DESCRIPTION, make_draft, make_product


def fields_with_issues(draft, product) -> set[str]:
    return {issue.field for issue in check_draft(draft, product)}


def test_clean_draft_passes_all_checks():
    assert check_draft(make_draft(), make_product()) == []


def test_numbers_are_compared_in_canonical_form():
    assert extract_numbers("0–1,6 МПа, DM 02").keys() == extract_numbers("от 0 до 1.6 МПа, DM 2").keys()


def test_invented_numbers_are_flagged():
    draft = make_draft(
        seo_description=GOOD_DESCRIPTION.replace("класс точности 1,5.", "класс точности 1,5, степень защиты IP65."),
        search_queries=["манометр dm 02", "манометр 16 бар", "манометр манотомь"],
    )

    unknown = find_unknown_numbers(draft, make_product())

    assert unknown == {"seo_description": ["65"], "search_queries": ["16"]}
    issues = [issue for issue in check_draft(draft, make_product()) if issue.field == "seo_description"]
    assert "65" in issues[0].problem and "65" in issues[0].fix_hint


def test_b2b_is_not_treated_as_a_number():
    draft = make_draft(commercial_block={"target_audience": ["B2B-закупки КИП"], "tasks_solved": ["Контроль давления"]})
    assert find_unknown_numbers(draft, make_product()) == {}


def test_extracted_spec_needs_verbatim_quote():
    draft = make_draft(specs_from_description=[
        {"name": "Измеряемая среда", "value": "газы и пар", "quote": "ГАЗОВ  И ПАРА"},
        {"name": "Материал корпуса", "value": "нержавеющая сталь", "quote": "корпус из нержавеющей стали"},
        {"name": "Температура среды", "value": "до 150 °C", "quote": "жидкостей, газов и пара"},
    ])

    verified, rejected = split_extracted_specs(draft, make_product())

    assert [spec.name for spec in verified] == ["Измеряемая среда"]
    assert [spec.name for spec in rejected] == ["Материал корпуса", "Температура среды"]
    assert "specs_from_description" in fields_with_issues(draft, make_product())


def test_extracted_specs_are_rejected_without_description():
    product = make_product(description=None)
    _, rejected = split_extracted_specs(make_draft(), product)
    assert len(rejected) == 1


def test_title_length_limit():
    product = make_product()
    assert "seo_title" not in fields_with_issues(make_draft(seo_title="М" * 70), product)
    assert "seo_title" in fields_with_issues(make_draft(seo_title="М" * 71), product)


def test_description_length_limits():
    product = make_product()
    base = GOOD_DESCRIPTION[:700]
    for length, has_issue in [(799, True), (800, False), (1200, False), (1201, True)]:
        text = (base + " продолжение текста" * 40)[:length]
        assert ("seo_description" in fields_with_issues(make_draft(seo_description=text), product)) is has_issue


def test_search_queries_count():
    product = make_product()
    assert "search_queries" in fields_with_issues(make_draft(search_queries=["манометр dm 02"]), product)
    six = [f"манометр dm 02 запрос {letter}" for letter in "абвгде"]
    assert "search_queries" in fields_with_issues(make_draft(search_queries=six), product)


def test_no_data_marker_in_public_text_is_flagged():
    draft = make_draft(seo_description=GOOD_DESCRIPTION + " Степень защиты: данных нет.")
    assert "seo_description" in fields_with_issues(draft, make_product())


def test_input_gaps_are_described():
    product = make_product(manufacturer="", description=None, specs={"класс точности": "1,5", "IP": ""})
    gaps = describe_input_gaps(product)
    assert len(gaps) == 2
    assert "производитель" in gaps[0]
    assert "description" in gaps[1]

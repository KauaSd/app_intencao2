import json
import pathlib

import pytest
import spacy

from intent.config import Settings
from intent.matching import UNKNOWN_INTENT, Matcher, rank
from intent.normalize import normalise
from intent.rules import RuleType, load_all_rulesets, load_ruleset, strip_accents

CASES = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "pt_cases.json")
    .read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        return spacy.blank("pt")


@pytest.fixture(scope="module")
def engine(nlp):
    ruleset = load_ruleset("pt")
    settings = Settings.from_env(env={})
    matcher = Matcher(ruleset, settings)
    return lambda text: rank(matcher.evaluate(nlp(text), normalise(text)), settings)


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:34] for c in CASES])
def test_pt_fixture(engine, case):
    result = engine(case["text"])
    assert result.primary == case["expected"], (
        f"{case['text']!r} -> got {result.primary}; "
        f"scores: {[(s.intent_id, s.score) for s in result.dropped]}")
    for other in case.get("also_expected", []):
        assert other in [m.intent for m in result.kept]
        assert result.multi_intent is True


def test_every_catalogued_intent_has_rules_and_examples():
    ruleset = load_ruleset("pt")
    assert {i.id for i in ruleset.intents} == {
        "cancelar", "reclamacao", "pagamento", "acesso", "upgrade", "comprar",
        "downgrade", "suporte", "alterar_dados", "duvida", "saudacao"}
    for intent in ruleset.intents:
        assert intent.rules, intent.id
        assert intent.examples, intent.id


def test_priorities_match_the_spec_catalogue():
    expected = {"cancelar": 90, "reclamacao": 80, "pagamento": 75, "acesso": 70,
                "upgrade": 65, "comprar": 60, "downgrade": 55, "suporte": 50,
                "alterar_dados": 45, "duvida": 40, "saudacao": 20}
    for intent in load_ruleset("pt").intents:
        assert intent.priority == expected[intent.id], intent.id


def test_pt_and_en_expose_the_same_intent_ids():
    assert ({i.id for i in load_ruleset("pt").intents}
            == {i.id for i in load_ruleset("en").intents})


def test_comprar_does_not_fire_on_desassinar(nlp):
    settings = Settings.from_env(env={})
    matcher = Matcher(load_ruleset("pt"), settings)
    result = rank(matcher.evaluate(nlp("quero desassinar o contrato"),
                                   normalise("quero desassinar o contrato")),
                  settings)
    assert "comprar" not in [m.intent for m in result.kept]


@pytest.mark.parametrize("language", ["pt", "en"])
def test_all_rules_name_two_distinct_signals(language):
    ruleset = load_ruleset(language)
    for intent in ruleset.intents:
        for rule in intent.rules:
            if rule.type is not RuleType.ALL:
                continue
            folded = [strip_accents(v.lower()) for v in rule.value]
            assert len(set(folded)) == len(folded), f"{rule.rule_id}: {rule.value}"


def test_default_path_loads_the_shipped_ruleset():
    ruleset = load_ruleset("pt")
    assert ruleset.language == "pt"
    assert ruleset.intents


def test_load_all_rulesets_loads_every_configured_language():
    rulesets = load_all_rulesets(("pt", "en"))
    assert set(rulesets) == {"pt", "en"}
    assert all(r.intents for r in rulesets.values())
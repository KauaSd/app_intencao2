import json
import pathlib

import pytest
from intent import rules
from intent.errors import RulesError, UnknownIntentError, UnknownLanguageError
from intent.rules import load_all_rulesets, load_ruleset, validate_document

VALID = {
    "language": "pt", "version": "1.0.0",
    "intents": [{
        "id": "cancelar", "label": "Cancelar assinatura", "priority": 90,
        "min_score": 3.0,
        "rules": [{"type": "keyword", "value": "cancelar", "weight": 2.0}],
        "examples": ["quero cancelar"],
    }],
    "negators": ["não", "deixar de"],
    "negation_boundaries": ["mas", "porque", "e"],
}


def _write(document, tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _load(document, tmp_path):
    return load_ruleset("pt", path=_write(document, tmp_path))


def test_valid_document_passes():
    validate_document(VALID)


@pytest.mark.parametrize("mutate,path", [
    (lambda d: d.update(language="pt-BR"), "language"),
    (lambda d: d.pop("language"), "language"),
    (lambda d: d["intents"][0]["rules"].append({"type": "nope", "value": "x", "weight": 1.0}),
     "intents.0.rules.1.type"),
    (lambda d: d["intents"][0]["rules"][0].update(value="duas palavras"),
     "intents.0.rules.0.value"),
    (lambda d: d["intents"][0]["rules"][0].update(value="[unclosed"),
     "intents.0.rules.0.value"),
    (lambda d: d["intents"][0]["rules"][0].update(weight=0),
     "intents.0.rules.0.weight"),
    (lambda d: d["intents"][0]["rules"][0].update(weight="two"),
     "intents.0.rules.0.weight"),
    (lambda d: d["intents"][0]["rules"][0].update(weight=float("inf")),
     "intents.0.rules.0.weight"),
    (lambda d: d["intents"][0].update(min_score=0), "intents.0.min_score"),
    (lambda d: d["intents"][0].update(id="Cancelar"), "intents.0.id"),
    (lambda d: d["intents"][0].update(priority="alta"), "intents.0.priority"),
    (lambda d: d["intents"].append(dict(d["intents"][0])), "intents.1.id"),
    (lambda d: d.update(version="1.0"), "version"),
    (lambda d: d["intents"][0]["rules"].append(
        {"type": "all", "value": ["so"], "weight": 1.0}), "intents.0.rules.1.value"),
    (lambda d: d["intents"][0]["rules"].append(
        {"type": "all", "value": ["a", "b c"], "weight": 1.0}), "intents.0.rules.1.value"),
    (lambda d: d["intents"][0]["rules"].append(
        dict(d["intents"][0]["rules"][0])), "intents.0.rules.1"),
    (lambda d: d.update(intents=[]), "intents"),
])
def test_invalid_documents_are_rejected_with_a_path(mutate, path):
    document = json.loads(json.dumps(VALID))
    mutate(document)
    with pytest.raises(RulesError) as exc:
        validate_document(document)
    assert exc.value.details["path"].startswith(path), exc.value.details


def test_nan_weight_is_rejected():
    # nan <= 0 is False, so a naive positivity check lets it through and it
    # then poisons every ranking
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"][0]["weight"] = float("nan")
    with pytest.raises(RulesError):
        validate_document(document)


def test_integer_weight_is_accepted(tmp_path):
    # Review Focus #5: these files are hand-edited by non-Python speakers
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"][0]["weight"] = 2
    validate_document(document)
    rule = _load(document, tmp_path).intents[0].rules[0]
    assert rule.weight == 2.0
    assert isinstance(rule.weight, float)


def test_boolean_weight_is_rejected(tmp_path):
    # float(True) is 1.0, and a `true` in the file is a typo, not a weight of
    # one point
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"][0]["weight"] = True
    with pytest.raises(RulesError) as exc:
        _load(document, tmp_path)
    assert exc.value.details["path"] == "intents.0.rules.0.weight"


def test_numeric_string_weight_is_accepted(tmp_path):
    # the same situation as a hand-written `2`: float() is what turns the number
    # the file carries into a score, whether it arrived as a number or as text
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"][0]["weight"] = "2"
    rule = _load(document, tmp_path).intents[0].rules[0]
    assert rule.weight == 2.0
    assert isinstance(rule.weight, float)


def test_duplicate_type_and_value_within_one_intent_is_rejected():
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"].append(
        {"type": "keyword", "value": "cancelar", "weight": 1.0})
    with pytest.raises(RulesError):
        validate_document(document)


def test_the_same_value_in_two_different_intents_is_fine(tmp_path):
    document = json.loads(json.dumps(VALID))
    other = dict(document["intents"][0])
    other["id"] = "comprar"
    document["intents"].append(other)
    validate_document(document)
    assert len(_load(document, tmp_path).intents) == 2


def _with_pattern(value, tmp_path):
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"] = [{"type": "pattern", "value": value, "weight": 3.0}]
    return _load(document, tmp_path)


def test_pattern_is_compiled_with_implicit_word_boundaries(tmp_path):
    rule = _with_pattern("assinar", tmp_path).intents[0].rules[0]
    assert rule.compiled.search("vou assinar o plano") is not None
    # the point: it must not fire inside a longer word
    assert rule.compiled.search("vou desassinar o plano") is None


def test_pattern_value_is_accent_stripped_at_load(tmp_path):
    rule = _with_pattern(r"não\s+funciona", tmp_path).intents[0].rules[0]
    assert rule.compiled.search("nao funciona") is not None


def test_all_value_is_a_tuple_and_never_compiled(tmp_path):
    # `all` is the only type whose value is a list in the file, and the list
    # becomes a tuple: the RuleSet is frozen and a mutable value would slip past
    # that contract
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"] = [
        {"type": "all", "value": ["cancelar", "assinatura"], "weight": 3.5}]
    rule = _load(document, tmp_path).intents[0].rules[0]
    assert rule.value == ("cancelar", "assinatura")
    assert isinstance(rule.value, tuple)
    assert rule.compiled is None


def test_rule_ids_are_indexed_by_position(tmp_path):
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"] = [
        {"type": "keyword", "value": "cancelar", "weight": 2.0},
        {"type": "lemma", "value": "cancelamento", "weight": 2.0},
    ]
    ruleset = _load(document, tmp_path)
    assert [r.rule_id for r in ruleset.intents[0].rules] == ["cancelar:0", "cancelar:1"]


def test_by_id_raises_for_an_unknown_intent(tmp_path):
    with pytest.raises(UnknownIntentError):
        _load(VALID, tmp_path).by_id("nao_existe")


def test_by_id_returns_the_spec(tmp_path):
    ruleset = _load(VALID, tmp_path)
    assert ruleset.by_id("cancelar").priority == 90


def test_missing_file_raises_unknown_language(tmp_path):
    with pytest.raises(UnknownLanguageError):
        load_ruleset("pt", path=tmp_path / "missing.json")


def test_default_path_loads_the_shipped_ruleset():
    ruleset = load_ruleset("pt")
    assert ruleset.language == "pt"
    assert ruleset.intents
    expected = pathlib.Path(rules.__file__).resolve().parent / "rules" / "pt.json"
    assert expected.is_file()
    assert "resources" not in str(expected)


def test_load_all_rulesets_loads_every_configured_language():
    rulesets = load_all_rulesets(("pt", "en"))
    assert set(rulesets) == {"pt", "en"}
    assert all(r.intents for r in rulesets.values())


def test_sha256_is_stable_and_content_dependent(tmp_path):
    a = _load(VALID, tmp_path)
    b = _load(json.loads(json.dumps(VALID)), tmp_path)
    c = _with_pattern("x", tmp_path)
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


def test_negators_and_boundaries_are_loaded(tmp_path):
    ruleset = _load(VALID, tmp_path)
    assert "deixar de" in ruleset.negators
    assert ruleset.negation_boundaries == frozenset({"mas", "porque", "e"})


def test_pattern_that_only_compiles_bare_is_rejected():
    # `(?i)` compiles on its own and does not compile inside the word
    # boundaries, which is how the value would really be compiled
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"] = [
        {"type": "pattern", "value": "(?i)nao", "weight": 1.0}]
    with pytest.raises(RulesError) as exc:
        validate_document(document)
    assert exc.value.details["path"] == "intents.0.rules.0.value"


def test_sha256_reacts_to_a_whitespace_only_change(tmp_path):
    # the digest is over the raw bytes, so reformatting the file counts as a
    # change too: it is what answers "what actually got loaded?"
    document = json.loads(json.dumps(VALID))
    spaced = tmp_path / "spaced.json"
    spaced.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    assert load_ruleset("pt", path=spaced).sha256 != load_ruleset("pt", path=compact).sha256


def test_intents_must_be_a_list_not_a_string():
    # the wrong type is refused with RulesError, not with a TypeError from
    # iterating a string: whoever maintains the file needs the path, not a
    # traceback
    document = json.loads(json.dumps(VALID))
    document["intents"] = "cancelar"
    with pytest.raises(RulesError) as exc:
        validate_document(document)
    assert exc.value.details["path"] == "intents"


def test_negators_must_be_a_list_of_strings():
    document = json.loads(json.dumps(VALID))
    document["negators"] = {"não": True}
    with pytest.raises(RulesError) as exc:
        validate_document(document)
    assert exc.value.details["path"] == "negators"


def test_a_file_that_is_not_json_is_a_rules_error(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text("{nao é json", encoding="utf-8")
    with pytest.raises(RulesError) as exc:
        load_ruleset("pt", path=path)
    assert "rules.json" in str(exc.value)

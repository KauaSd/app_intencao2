import dataclasses
import pytest
from intent.schemas import (IntentMatch, IntentResult, LanguageCandidate,
                            LanguageInfo, ResourceReport, RuleView, TokenView)


def test_token_view_fields():
    tv = TokenView("cancelar", "cancelar", "VERB", False, False, True, False)
    assert (tv.text, tv.lemma, tv.pos) == ("cancelar", "cancelar", "VERB")
    assert tv.matched is True
    assert tv.negated is False


def test_token_view_has_no_leaked_spacy_attributes():
    names = {f.name for f in dataclasses.fields(TokenView)}
    assert not any(n.endswith("_") for n in names)
    assert "sentiment" not in names


def test_token_view_is_frozen():
    tv = TokenView("a", "a", "NOUN", False, True, False, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tv.text = "b"


def test_intent_match_carries_min_score():
    # the UI renders "score / min_score", so the field has to exist here
    m = IntentMatch("cancelar", "Cancelar assinatura", 3.5, 0.78, 90, 3.0, ())
    assert m.min_score == 3.0


def test_language_candidate_shape():
    c = LanguageCandidate("pt", 0.8, ("quero", "quero"))
    assert c.language == "pt" and c.markers == ("quero", "quero")


def test_language_info_rejection_carries_the_diagnostics():
    info = LanguageInfo(None, False, "ambiguous_language", "Latin", True,
                        (LanguageCandidate("pt", 0.5, ()),
                         LanguageCandidate("en", 0.48, ())), 0.55, 0.02)
    assert info.margin == 0.02
    assert len(info.candidates) == 2


def test_resource_report_shape():
    r = ResourceReport({"pt": {"model": "blank(pt)", "fallback": True}},
                       {"pt": {"version": "1.0.0", "entries": 24, "sha256": "ab"}},
                       (), ())
    assert r.models["pt"]["fallback"] is True
    assert r.rules["pt"]["entries"] == 24


def test_rule_view_has_the_documented_fields():
    names = {f.name for f in dataclasses.fields(RuleView)}
    assert names == {"rule_id", "type", "value", "weight", "char_start",
                     "char_end", "negated", "matched_text", "offsets_exact"}


def test_intent_result_field_order():
    assert [f.name for f in dataclasses.fields(IntentResult)] == [
        "text", "normalised", "intents", "primary", "multi_intent", "confidence",
        "tokens", "language", "explanation", "rules_version", "trace",
        "resources", "duration_ms"]

import pytest

from intent.config import Settings
from intent.errors import InputError
from intent.matching import (UNKNOWN_INTENT, IntentScore, RawHit, confidence_for,
                             rank)
from intent.pipeline import Pipeline, explain
from intent.rules import RuleType


def _score(intent_id, score, priority, min_score=1.0):
    return IntentScore(intent_id, intent_id, score, priority, min_score, ())


def test_confidence_curve():
    assert confidence_for(0.0) == 0.0
    assert confidence_for(1.0) == 0.5
    assert confidence_for(3.0) == 0.75


def test_below_min_score_is_dropped(settings):
    r = rank([_score("comprar", 1.5, 60, min_score=2.0)], settings)
    assert r.kept == ()
    assert r.primary == UNKNOWN_INTENT
    assert r.dropped[0].intent_id == "comprar"


def test_a_tie_is_broken_by_priority(settings):
    r = rank([_score("suporte", 3.0, 50), _score("cancelar", 3.0, 90)], settings)
    assert r.primary == "cancelar"


def test_equal_priority_is_broken_by_id_deterministically(settings):
    forward = rank([_score("suporte", 3.0, 50), _score("acesso", 3.0, 50)], settings)
    reverse = rank([_score("acesso", 3.0, 50), _score("suporte", 3.0, 50)], settings)
    assert forward.primary == "acesso"
    assert reverse.primary == forward.primary


def test_multi_intent_is_reported_and_capped(settings):
    scores = [_score("cancelar", 5.0, 90), _score("comprar", 4.0, 60),
              _score("suporte", 3.0, 50), _score("duvida", 2.5, 40)]
    r = rank(scores, settings)
    assert r.multi_intent is True
    assert len(r.kept) == 3          # INTENT_MAX_INTENTS
    assert [m.intent for m in r.kept] == ["cancelar", "comprar", "suporte"]


def test_a_single_intent_is_not_multi(settings):
    assert rank([_score("cancelar", 5.0, 90)], settings).multi_intent is False


def test_the_confidence_gate_produces_desconhecido_but_keeps_the_evidence(settings):
    r = rank([_score("duvida", 0.4, 40, min_score=0.1)], settings)
    assert r.primary == UNKNOWN_INTENT
    assert r.kept == ()
    assert r.reason == "low_confidence"
    assert r.dropped          # the near-miss stays visible to the UI


def test_no_input_produces_desconhecido(settings):
    r = rank([], settings)
    assert r.primary == UNKNOWN_INTENT
    assert r.confidence == 0.0
    assert r.multi_intent is False
    assert r.reason == "ok"


def test_kept_matches_carry_min_score(settings):
    r = rank([_score("cancelar", 4.0, 90, min_score=3.0)], settings)
    assert r.kept[0].min_score == 3.0
    assert r.kept[0].confidence == round(4.0 / 5.0, 4)


# --------------------------------------------------------------------------
# the brief pins the eight behaviours above and is silent on the seven below.
# each one is a decision `rank` had to make, and each is written so that
# deleting the decision fails the test.
# --------------------------------------------------------------------------


def test_the_gate_does_not_promote_the_runner_up():
    # the brief's gate test has a single candidate, so a `rank` that answered
    # with the second-placed intent when the first failed the gate would pass
    # every test the brief ships. Here the winner is refused (0.4 -> 0.2857 <
    # 0.3) and a runner-up is present and would be kept by such a mutant.
    #
    # the numbers are constrained, and that is the point: `confidence_for` is
    # strictly increasing in `score` and the ranking leads with `score`, so a
    # refused winner is by construction the top scorer and every runner-up has
    # a *lower* confidence than it. Both entries therefore sit below the gate,
    # which is what makes "não confiei em nada" the only possible reading — and
    # it is also why promotion is not a subtle risk here but a plain one: it
    # would hand the caller an intent with a confidence it just rejected.
    settings = Settings.from_env(env={})
    r = rank([_score("cancelar", 0.4, 90, min_score=0.1),
              _score("suporte", 0.3, 50, min_score=0.1)], settings)
    assert r.primary == UNKNOWN_INTENT
    assert r.kept == ()
    assert r.confidence == 0.0
    assert r.reason == "low_confidence"
    # the refused winner is the first thing the trace has to show
    assert [s.intent_id for s in r.dropped] == ["cancelar", "suporte"]


def test_a_fully_negated_intent_is_not_a_candidate():
    # Task 6 returns a wholly negated intent with `score == 0.0` instead of
    # omitting it. `rules.py` refuses `min_score <= 0` at load, so a shipped
    # ruleset can never reach that case; a hand-built `IntentScore` can, and
    # `min_score == 0.0` is exactly what would let "não quero cancelar" be
    # reported as `cancelar`.
    settings = Settings.from_env(env={})
    r = rank([_score("cancelar", 0.0, 90, min_score=0.0)], settings)
    assert r.kept == ()
    assert r.primary == UNKNOWN_INTENT
    assert r.reason == "ok"          # nothing claimed it, so nothing refused it
    assert [s.intent_id for s in r.dropped] == ["cancelar"]


def test_the_zero_score_intent_loses_to_nothing_and_wins_over_nothing():
    # the same guard seen from the other side: a zero-score intent cannot
    # become the `primary` even when it is the only entry, and it cannot
    # displace a real one.
    settings = Settings.from_env(env={})
    r = rank([_score("cancelar", 0.0, 90, min_score=0.0),
              _score("suporte", 2.0, 50, min_score=1.0)], settings)
    assert r.primary == "suporte"
    assert [m.intent for m in r.kept] == ["suporte"]


def test_the_cap_keeps_what_it_cut_visible():
    # `RankedResult.dropped` is the only channel the diagnostic has, and an
    # intent that cleared its own `min_score` and lost the cap is the most
    # useful thing a reviewer can be shown. Both are the same total order, so
    # the list does not depend on the order the intents matched in.
    settings = Settings.from_env(env={})
    scores = [_score("duvida", 2.5, 40), _score("cancelar", 5.0, 90),
              _score("comprar", 4.0, 60), _score("suporte", 3.0, 50)]
    r = rank(scores, settings)
    assert [m.intent for m in r.kept] == ["cancelar", "comprar", "suporte"]
    assert [s.intent_id for s in r.dropped] == ["duvida"]


def test_below_min_score_stays_visible_next_to_the_answer():
    # a rejected intent is a near-miss the UI shows with its score; the answer
    # and the near-misses are reported by the same call.
    settings = Settings.from_env(env={})
    r = rank([_score("cancelar", 5.0, 90, min_score=3.0),
              _score("duvida", 1.0, 40, min_score=2.5)], settings)
    assert r.primary == "cancelar"
    assert r.reason == "ok"
    assert [s.intent_id for s in r.dropped] == ["duvida"]


def test_the_top_confidence_is_the_winner_confidence():
    # `RankedResult.confidence` is the confidence of `primary`, computed the
    # same way as the one on every `kept` entry — not the score, and not a
    # second curve. `multi_intent` and `confidence` are the two fields the UI
    # reads next to the badge, so a divergence here is a visible lie.
    settings = Settings.from_env(env={})
    r = rank([_score("comprar", 3.0, 60), _score("cancelar", 6.0, 90)], settings)
    assert r.primary == "cancelar"
    assert r.confidence == confidence_for(6.0) == 0.8571
    assert r.kept[0].confidence == 0.8571
    assert r.kept[1].confidence == confidence_for(3.0) == 0.75
    assert r.multi_intent is True


def test_a_cap_of_zero_still_answers():
    # literally it makes `kept` empty while a candidate exists, which is the
    # one combination the engine must never produce, and it would do it with
    # `reason == "ok"` — the most misleading pair available. The cap is
    # floored at 1; `min_confidence = 1.0` is how a caller says "report
    # nothing", and it says it in the field that is meant for it.
    settings = Settings.from_env(env={"INTENT_MAX_INTENTS": "0"})
    r = rank([_score("cancelar", 5.0, 90), _score("comprar", 4.0, 60)], settings)
    assert r.primary == "cancelar"
    assert [m.intent for m in r.kept] == ["cancelar"]
    assert r.reason == "ok"
    assert [s.intent_id for s in r.dropped] == ["comprar"]


def test_kept_matches_carry_the_whole_trace_including_negated_hits():
    # spec §5.2: a negated match is kept in the trace with its weight shown. A
    # `rank` that rebuilt `rule_hits` from non-negated hits only would answer
    # "why" with a trace that had been edited into agreeing with the answer.
    settings = Settings.from_env(env={})
    hits = (
        RawHit("cancelar:0", RuleType.KEYWORD, "cancelar", 2.0, 6, 14,
               "cancelar", True, (2,)),
        RawHit("cancelar:1", RuleType.PATTERN, "quero\\s+cancelar", 3.0, 0, 14,
               "quero cancelar", True, (0, 2), negated=True),
    )
    score = IntentScore("cancelar", "Cancelar assinatura", 2.0, 90, 1.0, hits)
    r = rank([score], settings)
    match = r.kept[0]
    assert (match.intent, match.label) == ("cancelar", "Cancelar assinatura")
    assert (match.score, match.priority, match.min_score) == (2.0, 90, 1.0)
    assert [view.rule_id for view in match.rule_hits] == ["cancelar:0", "cancelar:1"]
    assert [view.negated for view in match.rule_hits] == [False, True]
    assert [view.type for view in match.rule_hits] == ["keyword", "pattern"]
    assert [view.weight for view in match.rule_hits] == [2.0, 3.0]
    assert [(view.char_start, view.char_end) for view in match.rule_hits] == [
        (6, 14), (0, 14)]
    assert [view.matched_text for view in match.rule_hits] == [
        "cancelar", "quero cancelar"]


# --------------------------------------------------------------------------
# Task 9: Pipeline tests
# --------------------------------------------------------------------------


def test_empty_and_short_text_are_client_errors(engine_with_blank):
    for text in ("", "   ", "a"):
        with pytest.raises(InputError):
            engine_with_blank.detect(text)


def test_punctuation_only_is_empty_text_not_an_error(engine_with_blank):
    # long enough to pass the length gate, with nothing to reason about
    r = engine_with_blank.detect("!!!")
    assert r.language.reason == "empty_text"
    assert r.primary == "desconhecido"
    assert r.intents == ()


def test_too_long_is_a_client_error(engine_with_blank):
    with pytest.raises(InputError):
        engine_with_blank.detect("a" * 5001)


def test_detect_returns_the_expected_intent(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar meu plano")
    assert r.primary == "cancelar"
    assert r.rules_version == "1.0.0"


def test_the_explanation_is_portuguese_and_names_the_rules(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar meu plano")
    assert r.explanation == explain(r)
    assert "cancelar" in r.explanation.lower()


def test_tokens_expose_clean_names(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar meu plano")
    assert r.tokens
    assert any(t.matched for t in r.tokens)
    assert not any(hasattr(t, "lemma_") for t in r.tokens)


def test_negated_tokens_are_marked(engine_with_blank):
    r = engine_with_blank.detect("não quero cancelar")
    assert any(t.negated for t in r.tokens)
    assert r.primary == "desconhecido"


def test_the_resource_report_names_the_model_in_use(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar")
    assert r.resources.models["pt"]["fallback"] is True
    assert r.resources.models["pt"]["model"] == "blank(pt)"


def test_the_resource_report_carries_the_rules_digest(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar")
    assert r.resources.rules["pt"]["version"] == "1.0.0"
    assert len(r.resources.rules["pt"]["sha256"]) == 64


def test_an_unsupported_language_is_reported_not_guessed(engine_with_blank):
    r = engine_with_blank.detect("취소해줘")
    assert r.language.supported is False
    assert r.language.reason == "unsupported_script"
    assert r.primary == "desconhecido"
    assert r.intents == ()


def test_batch_preserves_input_order(engine_with_blank):
    texts = ["quero cancelar meu plano", "취소해줘", "oi, tudo bem?",
             "quero assinar o plano pro"]
    assert [r.text for r in engine_with_blank.detect_batch(texts)] == texts


def test_batch_agrees_with_single_detection(engine_with_blank):
    texts = ["quero cancelar meu plano", "quero assinar o plano pro",
             "i want to cancel my subscription"]
    batch = engine_with_blank.detect_batch(texts)
    singles = [engine_with_blank.detect(t) for t in texts]
    assert [r.primary for r in batch] == [r.primary for r in singles]


def test_batch_over_the_limit_is_rejected(engine_with_blank):
    with pytest.raises(InputError):
        engine_with_blank.detect_batch(["oi"] * 501)


def test_batch_of_zero_is_fine(engine_with_blank):
    assert engine_with_blank.detect_batch([]) == []


def test_duration_is_measured(engine_with_blank):
    assert engine_with_blank.detect("quero cancelar").duration_ms >= 0.0

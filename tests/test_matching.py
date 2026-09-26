import pytest
import spacy


KEYWORD_CANCELAR = [{"type": "keyword", "value": "cancelar", "weight": 2.0}]


def test_keyword_matches_an_exact_token(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("quero cancelar o plano")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].token_indices == (1,)


def test_keyword_does_not_match_inside_a_longer_word(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("quero descancelar o plano") == []


def test_pattern_matches_across_punctuation(detect_with, blank_nlp):
    rules = [{"type": "pattern", "value": r"cancelar\s+(meu\s+)?plano", "weight": 3.0}]
    scores = detect_with(blank_nlp, rules)("quero cancelar, meu plano, hoje")
    assert scores[0].score == 3.0


def test_pattern_hits_map_to_a_token_span(detect_with, blank_nlp):
    rules = [{"type": "pattern", "value": r"nao\s+funciona", "weight": 2.5}]
    scores = detect_with(blank_nlp, rules)("o app nao funciona desde ontem")
    assert scores[0].hits[0].token_indices


def test_pattern_does_not_match_inside_a_longer_word(detect_with, blank_nlp):
    # Review Focus #2
    rules = [{"type": "pattern", "value": "assinar", "weight": 2.0}]
    assert detect_with(blank_nlp, rules)("vou desassinar o plano") == []


def test_all_requires_every_string(detect_with, blank_nlp):
    rules = [{"type": "all", "value": ["cancelar", "assinatura"], "weight": 3.5}]
    assert detect_with(blank_nlp, rules)("quero cancelar") == []
    assert detect_with(blank_nlp, rules)("quero cancelar a assinatura")[0].score == 3.5


def test_all_tokens_are_consumed_and_not_double_counted(detect_with, blank_nlp):
    # without consumption, one mention of "cancelar" would score 3.5 + 2.0
    rules = [{"type": "all", "value": ["cancelar", "assinatura"], "weight": 3.5},
             {"type": "keyword", "value": "cancelar", "weight": 2.0}]
    assert detect_with(blank_nlp, rules)("quero cancelar a assinatura")[0].score == 3.5


def test_a_pattern_over_an_all_word_is_dropped(detect_with, blank_nlp):
    # a conjunção pagou por "plano" e por "maior", então um padrão que só
    # repete as mesmas duas palavras não é evidência nova: 3.5, não 6.5
    rules = [{"type": "all", "value": ["plano", "maior"], "weight": 3.5},
             {"type": "pattern", "value": r"plano\s+maior", "weight": 3.0}]
    assert detect_with(blank_nlp, rules)("quero um plano maior")[0].score == 3.5


def test_consumption_does_not_depend_on_rule_order(detect_with, blank_nlp):
    # the same two rules in the opposite order must score identically, or the
    # result would depend on how the JSON happened to be written
    keyword_first = [{"type": "keyword", "value": "cancelar", "weight": 2.0},
                     {"type": "all", "value": ["cancelar", "assinatura"],
                      "weight": 3.5}]
    all_first = list(reversed(keyword_first))
    assert (detect_with(blank_nlp, keyword_first)("quero cancelar a assinatura")[0].score
            == detect_with(blank_nlp, all_first)("quero cancelar a assinatura")[0].score
            == 3.5)


def test_repetition_does_not_inflate_the_score(detect_with, blank_nlp):
    # Review Focus #1
    once = detect_with(blank_nlp, KEYWORD_CANCELAR)("quero cancelar")
    thrice = detect_with(blank_nlp, KEYWORD_CANCELAR)(
        "cancelar cancelar cancelar cancelar")
    assert once[0].score == 2.0
    assert thrice[0].score == 2.0
    # every occurrence is still visible in the trace
    assert len(thrice[0].hits) == 4


def test_an_intent_with_no_hits_is_absent(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("quero um café") == []


def test_hit_offsets_point_at_the_original_accented_text(detect_with, blank_nlp):
    rules = [{"type": "pattern", "value": r"nao\s+quero", "weight": 2.0}]
    text = "olá, não quero isso"
    hit = detect_with(blank_nlp, rules)(text)[0].hits[0]
    # slicing the ORIGINAL and expecting the accent is the only proof the
    # index map is actually being used
    assert text[hit.char_start:hit.char_end].lower() == "não quero"
    assert hit.offsets_exact is True


def test_lemma_rule_when_the_model_can_lemmatise(detect_with):
    try:
        nlp = spacy.load("pt_core_news_sm")
    except OSError:
        pytest.skip("pt_core_news_sm not installed")
    rules = [{"type": "lemma", "value": "cancelamento", "weight": 2.0}]
    assert detect_with(nlp, rules)("quero fazer um cancelamento")[0].score == 2.0


def test_lemma_rule_is_skipped_silently_without_a_lemmatiser(detect_with, blank_nlp):
    # a blank pipe has no lemmatiser; the rule is dropped, not an error
    rules = [{"type": "lemma", "value": "cancelamento", "weight": 2.0}]
    assert detect_with(blank_nlp, rules)("quero fazer um cancelamento") == []

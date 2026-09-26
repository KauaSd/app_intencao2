KEYWORD_CANCELAR = [{"type": "keyword", "value": "cancelar", "weight": 2.0}]
KEYWORD_COMPRAR = [{"type": "keyword", "value": "comprar", "weight": 2.0}]
PATTERN_CANCELAR = [{"type": "pattern", "value": r"cancelar", "weight": 3.0}]
PATTERN_SPACE_MAIOR = [{"type": "pattern", "value": r"\smaior", "weight": 2.0}]
ALL_CANCELAR_ASSINATURA = [
    {"type": "all", "value": ["cancelar", "assinatura"], "weight": 3.5}]


def test_a_negator_before_the_match_negates(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não quero cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


def test_negator_immediately_before_the_match(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("não cancelar")[0].score == 0.0


def test_a_negator_beyond_the_window_does_not_negate(detect_with, blank_nlp):
    # "não" is four tokens back and the window is three
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não quero mesmo assim cancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_negation_does_not_cross_a_clause_boundary(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não gostei, mas quero cancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_newline_is_a_clause_boundary(detect_with, blank_nlp):
    # without the newline boundary, "não" would be reached and Comprar negated
    scores = detect_with(blank_nlp, KEYWORD_COMPRAR)("não cancelar\nquero comprar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_a_carriage_return_is_a_clause_boundary(detect_with, blank_nlp):
    # the `\r` half of the same rule, next to the `\n` half on purpose. The negator
    # sits directly against the break, so the break is the only thing that can stop
    # the scan, and a lone `\r` reaches `_is_clause_break` as a token of its own:
    # spaCy neither strips it nor folds it into a `\r\n`.
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR, negators=["nao"])(
        "nao\rquero cancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_multi_word_negator(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("deixar de cancelar")[0].score == 0.0


def test_a_message_that_is_only_a_negator_does_not_error(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("não") == []


def test_negation_is_per_occurrence_not_per_intent(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)(
        "não quero cancelar, mas vou cancelar depois")
    assert scores[0].score == 2.0
    assert sum(1 for h in scores[0].hits if h.negated) == 1


def test_a_negated_rule_removes_its_whole_weight(detect_with, blank_nlp):
    rules = KEYWORD_CANCELAR + [
      {"type": "pattern", "value": r"quero\s+cancelar", "weight": 3.0}]
    scores = detect_with(blank_nlp, rules)("não quero cancelar")
    assert scores[0].score == 0.0
    assert all(h.negated for h in scores[0].hits)

def test_a_rule_that_spans_its_own_negator_is_not_negated(detect_with, blank_nlp):
    # `suporte`'s `pattern nao\s+(funciona|abre|carrega|responde)` exists to match
    # the negative statement *as* the signal. Its span starts on the negator, so the
    # leftward scan has nothing to find and it must survive intact. Written here
    # because it is the inverse of the test above and the pair pins the rule.
    # the brief wrapped this rule in a `suporte` intent object; `detect_with`
    # builds the intent around the rules it is given, so the wrapper is a
    # `RulesError` at load time. The rule, the message and both assertions are
    # the brief's.
    rules = [{"type": "pattern", "value": r"nao\s+(funciona|abre)", "weight": 2.5}]
    scores = detect_with(blank_nlp, rules)("o app não funciona")
    assert scores[0].score == 2.5
    assert all(h.negated is False for h in scores[0].hits)


def test_negators_are_configurable(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR,
                         negators=["sem querer"])("sem querer cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


# The three tests below exist because the two boundary tests above cannot fail on
# their own. In "não gostei, mas quero cancelar" and "não cancelar\nquero
# comprar" the negator sits four tokens to the left, and the window is three: the
# scan runs out of window before it reaches it, so both tests pass with the
# boundary logic deleted outright (verified by mutation). Each one here puts the
# negator *inside* the window, so only the stop condition it names can save the
# hit - and deleting any one of the three then fails a test.


def test_a_clause_word_stops_the_scan_even_within_the_window(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_COMPRAR)("não mas quero comprar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_a_comma_stops_the_scan_even_within_the_window(detect_with, blank_nlp):
    # the comma is hardcoded in matching.py precisely because a content file can
    # forget it, and a forgotten comma is cross-clause negation nobody hears about
    scores = detect_with(blank_nlp, KEYWORD_COMPRAR)("não, quero comprar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_a_newline_stops_the_scan_even_within_the_window(detect_with, blank_nlp):
    # two pasted messages; the tokenizer keeps the "\n" as its own token
    scores = detect_with(blank_nlp, KEYWORD_COMPRAR)("não\nquero comprar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_a_negator_wins_over_a_boundary_word_at_the_same_position(detect_with, blank_nlp):
    # The order of the two checks is the contract, and it needs a position where
    # both answers are yes to be observable: `e` is in the default boundaries and
    # is the last word of the negator. Negator first, or the scan stops on `e` and
    # the negation is lost.
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR, negators=["não e"])("não e cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


def test_an_accented_boundary_word_from_the_rules_file_is_recognised(detect_with, blank_nlp):
    # the shipped catalogues write "porém" and "então" with the accent. Both sides
    # of the comparison are folded, so a hand-written accented entry works and a
    # message carrying the accent is stopped by it.
    scores = detect_with(blank_nlp, KEYWORD_COMPRAR, boundaries=["porém"])(
        "não porém quero comprar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


# spaCy turns any whitespace run of two or more characters into a token of its
# own, so a doubled space and a tab both reach the scan as tokens. The set in 5.2
# is closed - `.` `,` `;` `:` `!` `?` and a line break - and neither of those is
# a line break. Stopping on any whitespace token turns a typo into a silent
# missed negation, in the one diagnostic the UI is built around.


def test_a_doubled_space_is_not_a_clause_boundary(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não  quero cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


def test_a_tab_is_not_a_clause_boundary(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não\tquero cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


# Two occurrences of the same rule are two claims, and negating one says nothing
# about the other. The components of an `all` are not two claims: `all
# ["cancelar","assinatura"]` is one claim written as a conjunction, so negating a
# component withdraws the conjunction. 3.5 would clear `cancelar`'s min_score of
# 3.0, which is how "não cancelar, mas assinatura" got classified as `cancelar`.


def test_a_half_negated_conjunction_scores_nothing(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, ALL_CANCELAR_ASSINATURA)(
        "não cancelar, mas assinatura")
    assert scores[0].score == 0.0
    assert sum(1 for hit in scores[0].hits if hit.negated) == 1


def test_an_un_negated_conjunction_still_scores(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, ALL_CANCELAR_ASSINATURA)(
        "quero cancelar a assinatura")
    assert scores[0].score == 3.5
    assert not any(hit.negated for hit in scores[0].hits)


def test_a_half_negated_pattern_still_scores(detect_with, blank_nlp):
    # the all-or-nothing rule is for `all` alone. A `pattern` is one claim about
    # one span, and two occurrences of it are two claims, so this is the guard
    # against the conjunction rule leaking across the rule types.
    scores = detect_with(blank_nlp, PATTERN_CANCELAR)(
        "não cancelar, mas vou cancelar depois")
    assert scores[0].score == 3.0
    assert sum(1 for hit in scores[0].hits if hit.negated) == 1


# A doubled space is its own spaCy token, so a two-word negator whose words are
# separated by one is no longer two consecutive tokens and the phrase never gets
# assembled. "deixar  de cancelar" came back as `cancelar` at full weight, and all
# four multi-word negators in the shipped catalogue are exposed to it. The scan
# now walks the lexical tokens only, so whitespace costs neither a window step nor
# a phrase. A line break is whitespace-only too, and stays a clause break - which
# is the fourth test below, on the phrase path where that carve-out is easiest to
# get wrong.


def test_a_doubled_space_inside_a_multi_word_negator_still_negates(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("deixar  de cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


def test_a_doubled_space_in_and_around_a_multi_word_negator_negates(detect_with, blank_nlp):
    # a doubled space inside the phrase and another one between the phrase and
    # the hit, and a negator that is not in the catalogue
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR, negators=["não pretendo"])(
        "não  pretendo  cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


def test_a_line_break_inside_a_multi_word_negator_still_stops_the_scan(
        detect_with, blank_nlp):
    # the carve-out that keeps `test_newline_is_a_clause_boundary` alive, on the
    # phrase path. A scan that skipped every whitespace-only token would drop the
    # "\n" between "pretendo" and "cancelar", assemble "não pretendo" out of the
    # two tokens on its left, and negate a clause the user had ended.
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR, negators=["não pretendo"])(
        "não pretendo\ncancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_the_window_still_bounds_the_scan_across_doubled_spaces(detect_with, blank_nlp):
    # "não" is four lexical tokens to the left and the window is three. If the
    # window were counted in `Doc` positions while whitespace stopped being walked,
    # the four gaps would buy four extra steps and this would negate.
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)(
        "não  quero  mesmo  assim  cancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_a_hit_that_opens_on_whitespace_still_meets_the_boundary_in_front(
        detect_with, blank_nlp):
    # not one of the four above: this is the branch the lexical sequence had to
    # grow. A pattern can open inside a whitespace token - here the second one,
    # Doc index 3 - so the start of the scan is a translation, not the raw index,
    # and the comma in front of the hit still has to stop it. A guard rather than
    # a red test: the old code got this message right too, because the skipped
    # token is transparent in both index spaces.
    scores = detect_with(blank_nlp, PATTERN_SPACE_MAIOR)("não  ,  maior")
    assert scores[0].hits[0].token_indices == (3, 4)
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False

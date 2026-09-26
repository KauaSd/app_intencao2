import pytest
from intent.config import Settings
from intent.language import LanguageIdentifier, Reason, load_profiles


@pytest.fixture
def identifier():
    return LanguageIdentifier(load_profiles(), Settings.from_env(env={}))


def test_clear_portuguese(identifier):
    info = identifier.identify("quero cancelar a minha assinatura por favor")
    assert info.language == "pt"
    assert info.supported is True
    assert info.reason == Reason.OK
    assert info.min_share is None


def test_clear_english(identifier):
    info = identifier.identify("i want to cancel my subscription please")
    assert info.language == "en"
    assert info.supported is True
    assert info.reason == Reason.OK


def test_korean_is_an_unsupported_script(identifier):
    info = identifier.identify("취소해줘")
    assert info.supported is False
    assert info.reason == Reason.UNSUPPORTED_SCRIPT
    assert info.language is None
    assert info.script == "CJK"


def test_cyrillic_is_an_unsupported_script(identifier):
    assert identifier.identify("хочу отменить подписку").reason == Reason.UNSUPPORTED_SCRIPT


def test_arabic_is_an_unsupported_script(identifier):
    assert identifier.identify("أريد إلغاء الاشتراك").reason == Reason.UNSUPPORTED_SCRIPT


def test_latin_gibberish_has_insufficient_signal(identifier):
    info = identifier.identify("xkqvw brtz plmq zzzz")
    assert info.supported is False
    assert info.reason == Reason.INSUFFICIENT_SIGNAL


def test_one_english_word_in_a_portuguese_message_is_not_ambiguous(identifier):
    info = identifier.identify("quero cancelar a assinatura porque o app nao "
                               "abre e o erro aparece sempre no meu celular")
    assert info.language == "pt"
    assert info.supported is True


def test_genuinely_mixed_is_ambiguous(identifier):
    # seven decision tokens split 4/3 is the shortest genuinely mixed message
    # that can both clear min_language_share (0.55) and trip
    # min_language_margin (0.15). With five tokens the shares are multiples of
    # 0.2, so the closest pair that still clears the floor is 0.6/0.4 - a gap
    # of 0.2, which is a decision, not an abstention
    info = identifier.identify("the quer quero the my the quer")
    assert info.supported is False
    assert info.reason == Reason.AMBIGUOUS_LANGUAGE
    assert info.margin is not None
    assert info.margin < Settings.from_env(env={}).min_language_margin


def test_input_with_no_cased_character_is_empty_text(identifier):
    # Review Focus #3: long enough to pass the length gate, useless to reason about
    for text in ("!!!", "   ", "...", "12345", "🙂"):
        info = identifier.identify(text)
        assert info.supported is False, text
        assert info.reason == Reason.EMPTY_TEXT, text


def test_candidates_are_reported_even_when_rejected(identifier):
    # a rejected verdict you cannot diagnose is as useless as a wrong one
    info = identifier.identify("the quer")
    assert info.candidates
    # 0.5/0.5 is a tie *below* the floor, and step 4 checks the share before
    # the gap. No other input in this file has candidates whose top share falls
    # under min_language_share, so this line is the whole coverage of that
    # branch of step 4
    assert info.reason == Reason.INSUFFICIENT_SIGNAL
    assert info.min_share is not None
    # the ordering the brief mandates: share descending, then language
    # ascending. Here the shares are exactly equal, so this pins the language
    # half of the key
    assert [c.language for c in info.candidates] == ["en", "pt"]


def test_candidates_are_ordered_by_share_before_language(identifier):
    # the other half of the same key, which the tie above cannot reach: PT 0.8
    # over EN 0.4, so only "share descending" can produce this order
    info = identifier.identify("quero cancelar a minha assinatura por favor no")
    assert [c.language for c in info.candidates] == ["pt", "en"]


def test_script_is_reported(identifier):
    assert identifier.identify("quero cancelar").script == "Latin"
    assert identifier.identify("취소해줘").script == "CJK"


def test_reason_values_are_the_published_strings():
    # the values are the contract the API publishes, not the attribute names:
    # renaming EMPTY_TEXT to empty_txt would leave every comparison above
    # passing and every caller broken
    assert Reason.OK == "ok"
    assert Reason.EMPTY_TEXT == "empty_text"
    assert Reason.INSUFFICIENT_SIGNAL == "insufficient_signal"
    assert Reason.AMBIGUOUS_LANGUAGE == "ambiguous_language"
    assert Reason.UNSUPPORTED_SCRIPT == "unsupported_script"


def test_offset_preserved_is_true_for_latin(identifier):
    assert identifier.identify("quero cancelar").offset_preserved is True


def test_shared_markers_are_not_evidence():
    # "plano" is in both shared lists; it must not create a false signal
    profiles = {"languages": {
        "pt": {"exclusive": ["quero"], "shared": ["plano"]},
        "en": {"exclusive": ["quero"], "shared": ["plano"]},
    }}
    ident = LanguageIdentifier(profiles, Settings.from_env(env={}))
    info = ident.identify("plano")
    assert info.supported is False
    assert info.reason == Reason.INSUFFICIENT_SIGNAL


def test_a_borrowed_word_below_the_mixed_ratio_is_not_even_a_candidate(identifier):
    # the only English signal in this Portuguese sentence is "no" (de + o), 0.1
    # of the decision tokens. Below INTENT_MIXED_LANGUAGE_RATIO it is a word,
    # not a competing language, so it does not even reach the report
    info = identifier.identify("quero cancelar a assinatura porque o app nao "
                               "abre e o erro aparece sempre no meu celular")
    assert [c.language for c in info.candidates] == ["pt"]
    assert info.language == "pt"


def test_emoji_do_not_count_against_the_latin_script(identifier):
    # 11 letters against 5 emoji: an identifier that let the emoji into the
    # script denominator would see 0.69 of the message as Latin and refuse a
    # sentence with nothing in it but a shrug
    info = identifier.identify("quero o plano 🙂🙂🙂🙂🙂")
    assert info.script == "Latin"
    assert info.supported is True
    assert info.language == "pt"


def test_a_latin_plurality_below_the_script_threshold_is_unsupported(identifier):
    # 9 Latin letters against 8 Cyrillic ones. Latin leads, and leading is not
    # the test: 0.53 is below INTENT_MIN_SCRIPT_SHARE, and no rule file covers
    # a message that is half Cyrillic either
    info = identifier.identify("quero como да нет все")
    assert info.supported is False
    assert info.reason == Reason.UNSUPPORTED_SCRIPT


def test_offset_preserved_is_false_when_whitespace_collapses(identifier):
    # the collapsed space stands for two characters of the original, so a span
    # that begins on it cannot be mapped back exactly
    assert identifier.identify("quero  cancelar").offset_preserved is False


def test_a_decomposed_accent_does_not_hide_the_language(identifier):
    # "ã" arriving as "a" + a combining mark is the same word, and the token is
    # "nao" either way - which is why a profile entry with an accent in it
    # would match the composed form and silently miss the decomposed one. The
    # mark that vanishes does shift the offsets, though
    info = identifier.identify("não")
    assert info.language == "pt"
    assert info.offset_preserved is True
    decomposed = identifier.identify("n\u0303ao")
    assert decomposed.language == "pt"
    assert decomposed.offset_preserved is False


# The minimums the brief spells out for `exclusive`, copied from its own
# backticked lists and checked word-for-word against them when this file was
# written. The PT list is 39 tokens but 38 distinct words: the brief writes
# "muito" twice. A profile is data, and no verdict in the suite can see a
# missing or misspelled marker, so the data needs its own coverage.
REQUIRED_EXCLUSIVE = {
    "pt": (
        "o", "a", "os", "as", "de", "do", "da", "dos", "das", "que", "e", "em",
        "para", "um", "uma", "com", "nao", "sim", "por", "mais", "como", "ao",
        "aos", "pelo", "sob", "entre", "depois", "antes", "quero", "preciso",
        "posso", "qual", "quando", "muito", "tambem", "ja", "na", "obrigado",
    ),
    "en": (
        "i", "the", "a", "an", "of", "to", "and", "is", "are", "was", "were",
        "for", "in", "on", "at", "my", "me", "want", "need", "can", "how",
        "what", "when", "where", "which", "not", "do", "does", "please",
    ),
}
# The only words the brief requires in *both* exclusive lists. Every other
# homograph was resolved to one language, because a word in both is evidence
# for neither and manufactures false ambiguity.
REQUIRED_IN_BOTH = frozenset({"a", "do"})


@pytest.mark.parametrize("language", sorted(REQUIRED_EXCLUSIVE))
def test_the_shipped_profile_keeps_every_required_marker(language):
    exclusive = load_profiles()["languages"][language]["exclusive"]
    missing = sorted(set(REQUIRED_EXCLUSIVE[language]) - set(exclusive))
    assert missing == [], f"{language}.exclusive lost required markers: {missing}"


def test_no_english_marker_leaked_into_the_portuguese_profile():
    # "for" is the word this caught once: an English function word typed into
    # the PT list, in the position where "por" belongs. It changes no verdict
    # in this file, so nothing else here would ever notice
    profiles = load_profiles()["languages"]
    in_both = set(profiles["pt"]["exclusive"]) & set(profiles["en"]["exclusive"])
    assert in_both == set(REQUIRED_IN_BOTH)

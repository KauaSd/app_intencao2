from intent.normalize import (has_cased_character, normalise, strip_accents,
                              to_original_offset)


def test_lowercases_and_collapses_whitespace():
    assert normalise("  QuERO   Cancelar  ").text == "quero cancelar"


def test_accents_are_stripped():
    assert normalise("não funciona").text == "nao funciona"


def test_sentence_punctuation_is_preserved_for_patterns():
    assert normalise("Não funciona!").text == "nao funciona!"


def test_index_map_length_always_matches_the_text():
    n = normalise("não quero cancelar")
    assert len(n.index_map) == len(n.text)


def test_offset_after_an_accented_character_maps_back_correctly():
    # "ã" becomes "a" without changing the length, so the map is the identity
    # here and this pins the plumbing, not a shift. The shift this engine really
    # produces comes from collapsing whitespace: see
    # test_index_map_is_not_the_identity_when_whitespace_collapses
    original = "não quero cancelar"
    n = normalise(original)
    assert to_original_offset(n, n.text.index("quero")) == original.index("quero")


def test_offset_after_several_accented_characters():
    original = "não, não, não quero cancelar"
    n = normalise(original)
    assert to_original_offset(n, n.text.index("quero")) == original.index("quero")


def test_offsets_are_exact_where_nothing_was_removed():
    original = "quero cancelar agora"
    n = normalise(original)
    for word in ("quero", "cancelar", "agora"):
        assert to_original_offset(n, n.text.index(word)) == original.index(word)


def test_index_map_is_identity_for_plain_ascii():
    original = "quero cancelar"
    assert normalise(original).index_map == tuple(range(len(original)))


def test_index_map_is_not_the_identity_when_whitespace_collapses():
    # two leading spaces and the run before "quero" are all gone, so a map that
    # did no work would be four characters short here
    original = "  NÃo   quero  cancelar!  "
    n = normalise(original)
    assert n.text.index("quero") == 4
    assert original.index("quero") == 8
    assert n.index_map[4] == 8
    assert n.index_map != tuple(range(len(n.text)))


def test_offset_clamps_instead_of_raising():
    n = normalise("oi")
    assert to_original_offset(n, 0) == 0
    assert to_original_offset(n, 99) == 2


def test_strip_accents_does_not_lowercase():
    assert strip_accents("Não") == "Nao"


def test_strip_accents_leaves_other_scripts_alone():
    assert strip_accents("취소해줘") == "취소해줘"


def test_strip_accents_works_per_character_not_per_string():
    # recomposing the whole string would fuse these two jamo into one syllable
    decomposed = "\u1100\u1161"
    assert strip_accents(decomposed) == decomposed
    assert normalise(decomposed).text == decomposed.lower()


def test_empty_and_blank_inputs():
    assert normalise("").text == ""
    assert normalise("   \n\t ").text == ""


def test_newlines_collapse_to_a_single_space():
    assert normalise("não\nquero\tcancelar").text == "nao quero cancelar"


def test_a_bare_combining_mark_does_not_split_a_whitespace_run():
    # the mark is not whitespace and emits nothing, so the two sides are one run
    assert normalise("a \u0303 b").text == "a b"


def test_has_cased_character():
    assert has_cased_character("abc") is True
    assert has_cased_character("não") is True
    assert has_cased_character("   ") is False
    assert has_cased_character("!!!") is False
    assert has_cased_character("12345") is False
    assert has_cased_character("🙂") is False
    assert has_cased_character("\u0303") is False

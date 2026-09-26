import pytest
from intent.config import Settings, resources_dir


def test_defaults_need_no_env():
    s = Settings.from_env(env={})
    assert s.languages == ("pt", "en")
    assert s.db_path == ""
    assert s.min_text_chars == 2
    assert s.max_text_chars == 5000
    assert s.negation_window == 3
    assert s.max_intents == 3
    assert s.min_confidence == 0.3
    assert s.min_script_share == 0.85
    assert s.min_language_share == 0.55
    assert s.min_language_margin == 0.15
    assert s.mixed_language_ratio == 0.25
    assert s.batch_max_items == 500
    assert s.log_level == "INFO"


def test_languages_are_parsed_and_stripped():
    assert Settings.from_env(env={"INTENT_LANGUAGES": " pt , en "}).languages == ("pt", "en")


def test_languages_are_lowercased_and_deduplicated():
    assert Settings.from_env(env={"INTENT_LANGUAGES": "PT, pt ,en"}).languages == ("pt", "en")


def test_float_env_written_without_a_decimal_point():
    assert Settings.from_env(env={"INTENT_MIN_CONFIDENCE": "0"}).min_confidence == 0.0
    assert Settings.from_env(env={"INTENT_MIN_LANGUAGE_MARGIN": "1"}).min_language_margin == 1.0


def test_int_env_must_be_an_int():
    with pytest.raises(ValueError, match="INTENT_NEGATION_WINDOW"):
        Settings.from_env(env={"INTENT_NEGATION_WINDOW": "tres"})


def test_empty_language_list_is_rejected():
    with pytest.raises(ValueError, match="INTENT_LANGUAGES"):
        Settings.from_env(env={"INTENT_LANGUAGES": " , "})


def test_resources_dir_holds_the_profiles_file():
    assert (resources_dir() / "language_profiles.json").is_file()

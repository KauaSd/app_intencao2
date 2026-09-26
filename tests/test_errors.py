import pytest
from intent.errors import (Errors, InputError, IntentError, ModelUnavailableError,
                           RulesError, StorageError, UnsupportedLanguageError,
                           UnknownLanguageError, UnknownIntentError)


def test_codes_are_the_spec_strings():
    assert Errors.INPUT_INVALID == "input_invalid"
    assert Errors.UNSUPPORTED_LANGUAGE == "unsupported_language"
    assert Errors.UNKNOWN_LANGUAGE == "unknown_language"
    assert Errors.UNKNOWN_INTENT == "unknown_intent"
    assert Errors.INVALID_RULES == "invalid_rules"
    assert Errors.MODEL_UNAVAILABLE == "model_unavailable"
    assert Errors.STORAGE_ERROR == "storage_error"
    assert Errors.INTERNAL == "internal"


@pytest.mark.parametrize("cls,code", [
    (InputError, "input_invalid"),
    (UnsupportedLanguageError, "unsupported_language"),
    (UnknownLanguageError, "unknown_language"),
    (UnknownIntentError, "unknown_intent"),
    (RulesError, "invalid_rules"),
    (ModelUnavailableError, "model_unavailable"),
    (StorageError, "storage_error"),
])
def test_subclasses_carry_their_code(cls, code):
    err = cls("mensagem em portugues", {"campo": "text"})
    assert isinstance(err, IntentError)
    assert err.code == code
    assert err.details == {"campo": "text"}


def test_details_default_to_an_empty_dict():
    assert InputError("x").details == {}


def test_retryable_only_for_transient_failures():
    assert ModelUnavailableError("x").retryable is True
    assert StorageError("x").retryable is True
    assert InputError("x").retryable is False
    assert RulesError("x").retryable is False


def test_every_subclass_defaults_to_its_own_code():
    assert InputError("x").code == Errors.INPUT_INVALID
    assert RulesError("x").code == Errors.INVALID_RULES

"""Códigos de erro estáveis e a hierarquia de exceções do motor."""

from __future__ import annotations

__all__ = [
    "Errors",
    "IntentError",
    "InputError",
    "InternalError",
    "ModelUnavailableError",
    "RulesError",
    "StorageError",
    "UnknownIntentError",
    "UnknownLanguageError",
    "UnsupportedLanguageError",
]


class Errors:
    """Códigos de erro do motor. São contrato público: a API os expõe."""

    INPUT_INVALID: str = "input_invalid"
    UNSUPPORTED_LANGUAGE: str = "unsupported_language"
    UNKNOWN_LANGUAGE: str = "unknown_language"
    UNKNOWN_INTENT: str = "unknown_intent"
    INVALID_RULES: str = "invalid_rules"
    MODEL_UNAVAILABLE: str = "model_unavailable"
    STORAGE_ERROR: str = "storage_error"
    INTERNAL: str = "internal"

    RETRYABLE: frozenset[str] = frozenset({MODEL_UNAVAILABLE, STORAGE_ERROR})


class IntentError(Exception):
    """Erro de domínio com código estável, mensagem em PT-BR e detalhes.

    A classe base recebe o código na mão; as concretas passam por `_CodedError`,
    que injeta o código delas: `InputError("mensagem inválida")`.
    """

    code: str

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details) if details is not None else {}
        self.retryable = code in Errors.RETRYABLE


class _CodedError(IntentError):
    """Erro cujo código vem da própria classe, então resta só declarar `code`."""

    code: str

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(self.code, message, details)


class InputError(_CodedError):
    code = Errors.INPUT_INVALID


class UnsupportedLanguageError(_CodedError):
    code = Errors.UNSUPPORTED_LANGUAGE


class UnknownLanguageError(_CodedError):
    code = Errors.UNKNOWN_LANGUAGE


class UnknownIntentError(_CodedError):
    code = Errors.UNKNOWN_INTENT


class RulesError(_CodedError):
    code = Errors.INVALID_RULES


class ModelUnavailableError(_CodedError):
    code = Errors.MODEL_UNAVAILABLE


class StorageError(_CodedError):
    code = Errors.STORAGE_ERROR


class InternalError(_CodedError):
    code = Errors.INTERNAL

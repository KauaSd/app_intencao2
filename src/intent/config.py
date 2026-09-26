"""Configuração do motor, lida do ambiente com o prefixo ``INTENT_``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Settings", "resources_dir"]


@dataclass(frozen=True)
class Settings:
    """Configuração imutável do motor. Todos os padrões funcionam sem `.env`."""

    languages: tuple[str, ...] = ("pt", "en")
    db_path: str = ""
    min_text_chars: int = 2
    max_text_chars: int = 5000
    negation_window: int = 3
    max_intents: int = 3
    min_confidence: float = 0.3
    min_script_share: float = 0.85
    min_language_share: float = 0.55
    min_language_margin: float = 0.15
    mixed_language_ratio: float = 0.25
    batch_max_items: int = 500
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Lê a configuração de `env` (ou de `os.environ` quando omitido).

        Levanta `ValueError` nomeando a variável quando o valor não converte.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        default = cls()
        return cls(
            languages=_languages(source, default.languages),
            db_path=_str(source, "INTENT_DB_PATH", default.db_path),
            min_text_chars=_int(source, "INTENT_MIN_TEXT_CHARS", default.min_text_chars),
            max_text_chars=_int(source, "INTENT_MAX_TEXT_CHARS", default.max_text_chars),
            negation_window=_int(source, "INTENT_NEGATION_WINDOW", default.negation_window),
            max_intents=_int(source, "INTENT_MAX_INTENTS", default.max_intents),
            min_confidence=_float(source, "INTENT_MIN_CONFIDENCE", default.min_confidence),
            min_script_share=_float(source, "INTENT_MIN_SCRIPT_SHARE", default.min_script_share),
            min_language_share=_float(
                source, "INTENT_MIN_LANGUAGE_SHARE", default.min_language_share),
            min_language_margin=_float(
                source, "INTENT_MIN_LANGUAGE_MARGIN", default.min_language_margin),
            mixed_language_ratio=_float(
                source, "INTENT_MIXED_LANGUAGE_RATIO", default.mixed_language_ratio),
            batch_max_items=_int(source, "INTENT_BATCH_MAX_ITEMS", default.batch_max_items),
            log_level=_str(source, "INTENT_LOG_LEVEL", default.log_level),
        )


def resources_dir() -> Path:
    """Diretório `resources/` da raiz do repositório."""
    return Path(__file__).resolve().parents[2] / "resources"


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    return default if raw is None else raw.strip()


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um número inteiro, recebido {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um número, recebido {raw!r}") from exc


def _languages(env: Mapping[str, str], default: tuple[str, ...]) -> tuple[str, ...]:
    raw = env.get("INTENT_LANGUAGES")
    if raw is None:
        return default
    # minúsculas e sem repetição: um código em maiúsculas nunca casa com um
    # idioma detectado, e um repetido faria o mesmo trabalho duas vezes
    languages = tuple(dict.fromkeys(
        part.strip().lower() for part in raw.split(",") if part.strip()
    ))
    if not languages:
        raise ValueError(
            "INTENT_LANGUAGES deve listar ao menos um idioma, por exemplo 'pt,en'"
        )
    return languages

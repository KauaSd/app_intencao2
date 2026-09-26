"""Objetos de valor da resposta de detecção.

`frozen=True` garante que os campos não sejam reatribuídos depois de construídos.
Não garante imutibilidade profunda: `ResourceReport.models` e `.rules` são dicts
comuns e podem ser alterados por dentro.

A ordem dos campos é contrato: `IntentResult` e `IntentMatch` são construídos
posicionalmente pelo pipeline e lidos na ordem pela API e pela UI.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "IntentMatch",
    "IntentResult",
    "LanguageCandidate",
    "LanguageInfo",
    "ResourceReport",
    "RuleView",
    "TokenView",
]


@dataclass(frozen=True)
class TokenView:
    """Token anotado, com os acertos de regra que o tokenizer não tem."""

    text: str
    lemma: str
    pos: str
    is_stop: bool
    is_sentence_start: bool
    matched: bool
    negated: bool


@dataclass(frozen=True)
class LanguageCandidate:
    """Candidato de idioma com sua participação e as marcas que o sustentam."""

    language: str
    share: float
    markers: tuple[str, ...]


@dataclass(frozen=True)
class LanguageInfo:
    """Veredito de idioma. `min_share`/`margin` só existem em rejeições."""

    language: str | None
    supported: bool
    reason: str
    script: str
    offset_preserved: bool
    candidates: tuple[LanguageCandidate, ...]
    min_share: float | None = None
    margin: float | None = None


@dataclass(frozen=True)
class RuleView:
    """Regra que disparou, com os deslocamentos no texto original."""

    rule_id: str
    type: str
    value: str
    weight: float
    char_start: int
    char_end: int
    negated: bool
    matched_text: str
    offsets_exact: bool


@dataclass(frozen=True)
class IntentMatch:
    """Intenção pontuada. A UI mostra `score / min_score`."""

    intent: str
    label: str
    score: float
    confidence: float
    priority: int
    min_score: float
    rule_hits: tuple[RuleView, ...]


@dataclass(frozen=True)
class ResourceReport:
    """O que o motor carregou de verdade, por idioma.

    `models[language]` é `{"model": str, "fallback": bool}` e o modelo vale
    `f"blank({language})"` quando o fallback sem modelo está em uso.
    `rules[language]` é `{"version": str, "entries": int, "sha256": str}`.
    """

    models: dict[str, dict[str, str | bool]]
    rules: dict[str, dict[str, str | int]]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IntentResult:
    """Resposta completa de uma detecção."""

    text: str
    normalised: str
    intents: tuple[IntentMatch, ...]
    primary: str
    multi_intent: bool
    confidence: float
    tokens: tuple[TokenView, ...]
    language: LanguageInfo
    explanation: str
    rules_version: str
    trace: tuple[RuleView, ...]
    resources: ResourceReport
    duration_ms: float

"""Use cases: detect, batch, catalogue, chat, health, history."""

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from intent.chat import Chatbot, ChatResponse
from intent.config import Settings
from intent.pipeline import Pipeline
from intent.rules import RuleSet
from intent.schemas import IntentResult, LanguageInfo


@dataclass
class IntentService:
    """Application use cases orchestrating the engine and chatbot."""

    pipeline: Pipeline
    chatbot: Chatbot
    db_path: str = ""

    def detect(self, text: str, require_supported: bool = False) -> IntentResult:
        """Single message detection."""
        return self.pipeline.detect(text)

    def detect_batch(
        self, texts: list[str], require_supported: bool = False
    ) -> list[IntentResult]:
        """Batch detection, grouped by language."""
        return self.pipeline.detect_batch(texts)

    def catalogue(self, language: Optional[str] = None) -> dict:
        """Return the intent catalogue with rule counts."""
        if language:
            ruleset = self.pipeline.rulesets.get(language)
            if not ruleset:
                return {"language": language, "intents": []}
            return self._ruleset_to_catalogue(ruleset)

        return {
            lang: self._ruleset_to_catalogue(ruleset)
            for lang, ruleset in self.pipeline.rulesets.items()
        }

    def _ruleset_to_catalogue(self, ruleset: RuleSet) -> dict:
        return {
            "language": ruleset.language,
            "version": ruleset.version,
            "intents": [
                {
                    "id": intent.id,
                    "label": intent.label,
                    "priority": intent.priority,
                    "min_score": intent.min_score,
                    "rule_count": len(intent.rules),
                }
                for intent in ruleset.intents
            ],
        }

    def rules(self, language: str) -> Optional[dict]:
        """Return active rules for a language."""
        ruleset = self.pipeline.rulesets.get(language)
        if not ruleset:
            return None
        return {
            "language": ruleset.language,
            "version": ruleset.version,
            "intents": [
                {
                    "id": intent.id,
                    "label": intent.label,
                    "priority": intent.priority,
                    "min_score": intent.min_score,
                    "rules": [
                        {
                            "rule_id": rule.rule_id,
                            "type": rule.type.value,
                            "value": rule.value,
                            "weight": rule.weight,
                        }
                        for rule in intent.rules
                    ],
                }
                for intent in ruleset.intents
            ],
            "negators": list(ruleset.negators),
            "negation_boundaries": list(ruleset.negation_boundaries),
        }

    def validate_rules(self, document: dict) -> tuple[bool, Optional[str]]:
        """Validate a rules document without activating it."""
        from intent.rules import validate_document
        try:
            validate_document(document)
            return True, None
        except Exception as e:
            return False, str(e)

    def health(self) -> dict:
        """Health check: models in use, fallbacks, limitations, rules versions."""
        models = {}
        rules = {}
        limitations = []
        warnings = []

        for lang, ruleset in self.pipeline.rulesets.items():
            rules[lang] = {
                "version": ruleset.version,
                "entries": len(ruleset.intents),
                "sha256": ruleset.sha256,
            }

            nlp = self.pipeline._nlp(lang)
            model_name = nlp.meta.get("name")
            fallback = model_name is None or model_name == "pipeline"
            if fallback:
                model_name = f"blank({lang})"
            models[lang] = {"model": model_name, "fallback": fallback}

            if fallback:
                limitations.append(f"Using blank pipeline for {lang} (model not installed)")

            has_lemma = nlp.has_pipe("lemmatizer")
            has_lemma_rules = any(
                r.type.value == "lemma"
                for intent in ruleset.intents
                for r in intent.rules
            )
            if has_lemma_rules and not has_lemma:
                limitations.append(
                    f"Lemma rules present for {lang} but pipeline cannot lemmatise"
                )

        return {
            "models": models,
            "rules": rules,
            "limitations": limitations,
            "warnings": warnings,
        }

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        require_supported: bool = False,
        language_hint: Optional[str] = None,
    ) -> ChatResponse:
        """Process a chat message through the chatbot and persist conversation."""
        response = self.chatbot.responder(message, session_id, self.pipeline)

        if self.db_path:
            try:
                self.store_conversation(
                    session_id=response.session_id,
                    turn_index=response.turn_index,
                    role="user",
                    text=message,
                    primary_intent=response.primary,
                    strategy=response.strategy,
                )
                self.store_conversation(
                    session_id=response.session_id,
                    turn_index=response.turn_index,
                    role="bot",
                    text=response.reply,
                    primary_intent=response.primary,
                    strategy=response.strategy,
                )
            except Exception:
                pass

        return response

    def history(self, limit: int = 20) -> list[dict]:
        """Return stored conversation summaries."""
        if not self.db_path:
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT session_id, turn_index, role, text, primary_intent, "
                "strategy, created_at FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception:
            return []

    def store_conversation(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        text: str,
        primary_intent: str,
        strategy: str,
    ) -> None:
        """Persist a conversation turn."""
        if not self.db_path:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT,
                    turn_index INTEGER,
                    role TEXT,
                    text TEXT,
                    primary_intent TEXT,
                    strategy TEXT,
                    created_at REAL
                )""",
            )
            conn.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, turn_index, role, text, primary_intent, strategy, time.time()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


def explain(result: IntentResult | LanguageInfo) -> str:
    """Generate PT-BR explanation from a detection result."""
    if isinstance(result, LanguageInfo):
        if not result.supported:
            return result.reason
        return ""

    if result.primary == "desconhecido":
        if result.language and not result.language.supported:
            return "Idioma não suportado."
        return "Nenhum intento identificado com confiança suficiente."

    match = result.intents[0] if result.intents else None
    if not match:
        return "Nenhum intento identificado."

    rule_names = []
    for hit in match.rule_hits:
        if not hit.negated:
            rule_names.append(f"{hit.type} '{hit.value}'")
    rules_str = ", ".join(rule_names) if rule_names else "nenhuma regra"
    return f"Mensagem classificada como {match.label} (regras: {rules_str})."
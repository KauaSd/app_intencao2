import time
from dataclasses import dataclass
from typing import Any, Optional

import spacy

from intent.config import Settings
from intent.errors import InputError
from intent.language import LanguageIdentifier, Reason
from intent.matching import Matcher, RankedResult, UNKNOWN_INTENT, confidence_for, rank
from intent.normalize import normalise
from intent.rules import RuleSet
from intent.schemas import (
    IntentMatch,
    IntentResult,
    LanguageInfo,
    ResourceReport,
    RuleView,
    TokenView,
)


@dataclass(frozen=True)
class Pipeline:
    rulesets: dict[str, RuleSet]
    profiles: dict
    settings: Settings
    nlp_by_language: Optional[dict[str, Any]] = None

    def _nlp(self, language: str):
        if self.nlp_by_language and language in self.nlp_by_language:
            return self.nlp_by_language[language]
        try:
            return spacy.load(f"{language}_core_news_sm")
        except OSError:
            nlp = spacy.blank(language)
            nlp.add_pipe("sentencizer")
            return nlp

    def _has_lemmatiser(self, language: str) -> bool:
        nlp = self._nlp(language)
        return nlp.has_pipe("lemmatizer")

    def detect(self, text: str) -> IntentResult:
        start = time.perf_counter()

        stripped = text.strip()
        if not stripped:
            raise InputError("empty text", {"min_chars": self.settings.min_text_chars})
        if len(stripped) < self.settings.min_text_chars:
            raise InputError(
                "text too short",
                {"min_chars": self.settings.min_text_chars, "actual": len(stripped)},
            )
        if len(stripped) > self.settings.max_text_chars:
            raise InputError(
                "text too long",
                {"max_chars": self.settings.max_text_chars, "actual": len(stripped)},
            )

        n = normalise(stripped)
        identifier = LanguageIdentifier(self.profiles, self.settings)
        info = identifier.identify(stripped)

        if not info.supported:
            result = IntentResult(
                text=stripped,
                normalised=n.text,
                intents=(),
                primary=UNKNOWN_INTENT,
                multi_intent=False,
                confidence=0.0,
                tokens=(),
                language=info,
                explanation=self._explain_unsupported(info),
                rules_version="",
                trace=(),
                resources=self._resource_report(info.language or "pt"),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return result

        doc = self._nlp(info.language)(stripped)
        matcher = Matcher(self.rulesets[info.language], self.settings)
        scores = matcher.evaluate(doc, n)
        result = rank(scores, self.settings)

        token_hit_indices = self._collect_token_indices(scores, result)
        tokens = self._build_tokens(doc, token_hit_indices)
        trace = self._build_trace(result)
        resources = self._resource_report(info.language)

        primary = result.primary
        confidence = result.confidence
        multi_intent = result.multi_intent
        intents = result.kept
        rules_version = self.rulesets[info.language].version

        result_obj = IntentResult(
            text=stripped,
            normalised=n.text,
            intents=intents,
            primary=primary,
            multi_intent=multi_intent,
            confidence=confidence,
            tokens=tokens,
            language=info,
            explanation=explain(IntentResult(
                text=stripped,
                normalised=n.text,
                intents=intents,
                primary=primary,
                multi_intent=multi_intent,
                confidence=confidence,
                tokens=tokens,
                language=info,
                explanation="",
                rules_version=rules_version,
                trace=trace,
                resources=resources,
                duration_ms=0.0,
            )),
            rules_version=rules_version,
            trace=trace,
            resources=resources,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        return result_obj

    def detect_batch(self, texts: list[str]) -> list[IntentResult]:
        if len(texts) > self.settings.batch_max_items:
            raise InputError(
                "batch too large",
                {"max_items": self.settings.batch_max_items, "actual": len(texts)},
            )

        results = [None] * len(texts)
        lang_groups = {}
        identifier = LanguageIdentifier(self.profiles, self.settings)

        for i, text in enumerate(texts):
            stripped = text.strip()
            n = normalise(stripped)
            info = identifier.identify(stripped)

            if not info.supported:
                results[i] = IntentResult(
                    text=stripped,
                    normalised=n.text,
                    intents=(),
                    primary=UNKNOWN_INTENT,
                    multi_intent=False,
                    confidence=0.0,
                    tokens=(),
                    language=info,
                    explanation=self._explain_unsupported(info),
                    rules_version="",
                    trace=(),
                    resources=self._resource_report(info.language or "pt"),
                    duration_ms=0.0,
                )
                continue

            lang = info.language
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append((i, stripped))

        for lang, items in lang_groups.items():
            nlp = self._nlp(lang)
            docs = nlp.pipe([t for _, t in items])
            matcher = Matcher(self.rulesets[lang], self.settings)
            for (i, text), doc in zip(items, docs):
                start = time.perf_counter()
                n = normalise(text)
                scores = matcher.evaluate(doc, n)
                result = rank(scores, self.settings)

                token_hit_indices = self._collect_token_indices(scores, result)
                tokens = self._build_tokens(doc, token_hit_indices)
                trace = self._build_trace(result)
                resources = self._resource_report(lang)

                primary = result.primary
                confidence = result.confidence
                multi_intent = result.multi_intent
                intents = result.kept
                rules_version = self.rulesets[lang].version

                temp_result = IntentResult(
                    text=text,
                    normalised=n.text,
                    intents=intents,
                    primary=primary,
                    multi_intent=multi_intent,
                    confidence=confidence,
                    tokens=tokens,
                    language=identifier.identify(text),
                    explanation="",
                    rules_version=rules_version,
                    trace=trace,
                    resources=resources,
                    duration_ms=0.0,
                )
                explanation = explain(temp_result)

                result_obj = IntentResult(
                    text=text,
                    normalised=n.text,
                    intents=intents,
                    primary=primary,
                    multi_intent=multi_intent,
                    confidence=confidence,
                    tokens=tokens,
                    language=identifier.identify(text),
                    explanation=explanation,
                    rules_version=rules_version,
                    trace=trace,
                    resources=resources,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                results[i] = result_obj

        return results

    def _build_tokens(self, doc, token_hit_indices: dict) -> tuple[TokenView, ...]:
        matched_token_indices = token_hit_indices.get("matched", set())
        negated_token_indices = token_hit_indices.get("negated", set())

        tokens = []
        for token in doc:
            tokens.append(
                TokenView(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    is_stop=token.is_stop,
                    is_sentence_start=token.is_sent_start,
                    matched=token.i in matched_token_indices,
                    negated=token.i in negated_token_indices,
                )
            )
        return tuple(tokens)

    def _collect_token_indices(self, scores: list, result: RankedResult) -> dict:
        matched = set()
        negated = set()
        kept_intent_ids = {m.intent for m in result.kept}

        for score in scores:
            if score.intent_id not in kept_intent_ids and result.primary != UNKNOWN_INTENT:
                continue
            for hit in score.hits:
                if hit.negated:
                    negated.update(hit.token_indices)
                else:
                    matched.update(hit.token_indices)

        return {"matched": matched, "negated": negated}

    def _build_trace(self, result: RankedResult) -> tuple[RuleView, ...]:
        trace = []
        for match in result.kept:
            for hit in match.rule_hits:
                trace.append(
                    RuleView(
                        rule_id=hit.rule_id,
                        type=hit.type,
                        value=hit.value,
                        weight=hit.weight,
                        char_start=hit.char_start,
                        char_end=hit.char_end,
                        negated=hit.negated,
                        matched_text=hit.matched_text,
                        offsets_exact=hit.offsets_exact,
                    )
                )
        for dropped in result.dropped:
            for hit in dropped.hits:
                trace.append(
                    RuleView(
                        rule_id=hit.rule_id,
                        type=hit.type.value,
                        value=hit.value,
                        weight=hit.weight,
                        char_start=hit.char_start,
                        char_end=hit.char_end,
                        negated=hit.negated,
                        matched_text=hit.matched_text,
                        offsets_exact=hit.offsets_exact,
                    )
                )
        return tuple(trace)

    def _resource_report(self, language: str) -> ResourceReport:
        models = {}
        rules = {}
        limitations = []
        warnings = []

        if language in self.rulesets:
            ruleset = self.rulesets[language]
            rules[language] = {
                "version": ruleset.version,
                "entries": len(ruleset.intents),
                "sha256": ruleset.sha256,
            }

        nlp = self._nlp(language)
        model_name = nlp.meta.get("name")
        fallback = model_name is None or model_name == "pipeline"
        if fallback:
            model_name = f"blank({language})"
        models[language] = {"model": model_name, "fallback": fallback}

        if fallback:
            limitations.append(f"Using blank pipeline for {language} (model not installed)")

        has_lemma = nlp.has_pipe("lemmatizer")
        if language in self.rulesets:
            has_lemma_rules = any(
                r.type.value == "lemma"
                for intent in self.rulesets[language].intents
                for r in intent.rules
            )
            if has_lemma_rules and not has_lemma:
                limitations.append(
                    f"Lemma rules present for {language} but pipeline cannot lemmatise"
                )

        return ResourceReport(
            models=models,
            rules=rules,
            limitations=tuple(limitations),
            warnings=tuple(warnings),
        )

    def _explain_unsupported(self, info: LanguageInfo) -> str:
        if info.reason == Reason.EMPTY_TEXT:
            return "Mensagem sem conteúdo utilizável para classificação."
        if info.reason == Reason.UNSUPPORTED_SCRIPT:
            return f"Idioma não suportado (escrita {info.script})."
        if info.reason == Reason.INSUFFICIENT_SIGNAL:
            return "Não há sinais linguísticos suficientes para identificar o idioma."
        if info.reason == Reason.AMBIGUOUS_LANGUAGE:
            return "Idioma ambíguo entre candidatos."
        return "Idioma não suportado."


def explain(result: RankedResult | IntentResult) -> str:
    if isinstance(result, IntentResult):
        if result.primary == UNKNOWN_INTENT:
            if result.language and not result.language.supported:
                return result.explanation
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
    return ""
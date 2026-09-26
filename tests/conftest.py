import json
import pathlib

import pytest
import spacy

from intent.config import Settings
from intent.language import load_profiles
from intent.matching import Matcher
from intent.normalize import normalise
from intent.pipeline import Pipeline
from intent.rules import load_all_rulesets, load_ruleset


@pytest.fixture
def settings():
    return Settings.from_env(env={})


@pytest.fixture(scope="session")
def blank_nlp():
    return spacy.blank("pt")


@pytest.fixture
def detect_with(tmp_path):
    """Build a Matcher over a throwaway one-intent ruleset and return a callable
    that evaluates one message. Used by test_matching and test_negation."""
    def make(nlp, rules, negators=None, boundaries=None, **env_overrides):
        document = {
            "language": "pt", "version": "1.0.0",
            "intents": [{"id": "cancelar", "label": "Cancelar assinatura",
                         "priority": 90, "min_score": 3.0, "rules": rules,
                         "examples": []}],
            "negators": negators if negators is not None else ["não", "deixar de"],
            "negation_boundaries": (boundaries if boundaries is not None
                                    else ["mas", "porque", "e"]),
        }
        path = tmp_path / "matcher.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        matcher = Matcher(load_ruleset("pt", path=path),
                          Settings.from_env(env=env_overrides))
        return lambda text: matcher.evaluate(nlp(text), normalise(text))
    return make


@pytest.fixture
def blank_pipes():
    return {"pt": spacy.blank("pt"), "en": spacy.blank("en")}


@pytest.fixture
def engine_with_blank(settings, blank_pipes):
    return Pipeline(rulesets=load_all_rulesets(settings.languages),
                    profiles=load_profiles(),
                    settings=settings,
                    nlp_by_language=blank_pipes)

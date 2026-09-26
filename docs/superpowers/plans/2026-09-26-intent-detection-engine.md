# Intent Detection Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic rules engine that classifies a chatbot message into a
priority-ranked intent (`cancelar`, `comprar`, `suporte`, …), exposed as a
library, a REST API, a CLI, a visual web UI, and a rule-based chatbot.

**Architecture:** Five layers, strictly one-directional:
`cli/api → service → pipeline → {matching → rules, normalize, language}`.
`normalize.py` owns the accent-stripped copy plus the index map that keeps
reported offsets pointing at the original text. `matching.py` evaluates four
declarative rule types, discards negated matches, and ranks by
`(score, priority, id)`. Rules are JSON content, not code. `service.py` holds no
scoring logic; `matching.py` touches neither filesystem nor settings, so every
dependency is constructor-injected and no test uses `monkeypatch` on a module
attribute.

**Tech Stack:** Python 3.14 · spaCy 3.8 (`pt_core_news_sm`, `en_core_web_sm`) ·
FastAPI · Uvicorn · Typer · Pydantic v2 · pytest · vanilla ES modules

**Spec:** `docs/superpowers/specs/2026-09-26-intent-detection-design.md` — the
plan argues from the spec, so read it alongside. Section numbers in parentheses
refer to it.

## Global Constraints

- Every user-facing string — UI copy, CLI output, `explanation`, chatbot
  replies — is **PT-BR**. Every code identifier, rule `id`, and API path is
  **English**. Never mix the two in one field.
- Settings come from env with prefix `INTENT_`, and these defaults must work
  with no env file at all. Never add a setting that nothing reads.
  - `INTENT_LANGUAGES=pt,en` · `INTENT_DB_PATH=""` (disabled) ·
    `INTENT_MIN_TEXT_CHARS=2` · `INTENT_MAX_TEXT_CHARS=5000` ·
    `INTENT_NEGATION_WINDOW=3` · `INTENT_MAX_INTENTS=3` ·
    `INTENT_MIN_CONFIDENCE=0.3` · `INTENT_MIN_SCRIPT_SHARE=0.85` ·
    `INTENT_MIN_LANGUAGE_SHARE=0.55` · `INTENT_MIN_LANGUAGE_MARGIN=0.15` ·
    `INTENT_MIXED_LANGUAGE_RATIO=0.25` · `INTENT_BATCH_MAX_ITEMS=500` ·
    `INTENT_LOG_LEVEL=INFO`
- spaCy attributes with trailing underscores (`token.lemma_`, `token.pos_`,
  `token.is_stop`) **never** cross a module boundary. `TokenView` exposes
  `lemma`, `pos`, `is_stop`, `matched`, `negated` (§9.5).
- No network access at runtime or in tests. Models load from local disk only.
- The UI uses ES modules from local files. No CDN, no build step, no `npm`. No
  `innerHTML` with message data — `textContent` and DOM construction only.
- Every test constructs its own dependencies. `monkeypatch` of module
  attributes is banned from the whole suite (§13). Patching `os.environ` to
  isolate configuration is fine and is the one exception.
- `git` is **not installed** here. Every "Commit" step is conditional: if
  `git --version` fails, note the change and move on. Do not install git, and do
  not let a commit failure block a task.
- Use `pytest <file>::<test>` while iterating; `pytest -v` at each task boundary.
  The first run is slow because it loads spaCy models.

## Review Focus

Five input classes the spec implies but no single task's tests naturally cover.
Each has a test in the task named beside it. Ordered by how expensive the
failure is.

1. **A word repeated to inflate the score.** `"cancelar cancelar cancelar"` must
   score exactly what `"cancelar"` scores (§5.3). Without per-rule dedupe it
   scores 8.0, and a user typing angrily crosses `cancelar`'s `min_score` twice
   over. Test in Task 5.
2. **A rule firing inside a longer word.** `pattern: "assinar"` must not match
   `desassinar` — someone unsubscribing would trigger `comprar` (§4.1). Test in
   Task 5, and again against the real rules file in Task 8.
3. **Input with no usable text but enough characters.** `"!!!"`, `"   "`,
   `"12345"` give `empty_text`, not `insufficient_signal` and not a crash in the
   profile counter (§5.1, §6.2). Test in Task 3.
4. **`require_supported: true` on a *supported* language with no intent.** Must
   be `200` with `desconhecido`, never `422` (§8). A client retrying on 422
   loops forever. Test in Task 10, and again at the HTTP boundary in Task 12.
5. **Hand-edited JSON with `"weight": 2` instead of `2.0`.** Must load and behave
   as `2.0`; these files are edited by people who do not write Python (§4.5).
   Test in Task 4.

---

## File Map

Every file and the one thing it owns. A file not listed here does not exist.

```
aa/
  pyproject.toml                 setuptools, src layout, deps, pytest config
  Makefile                       install / test / demo / rules-validate / serve
  README.md  .env.example  .gitignore
  app.py                         uvicorn entrypoint, delegates to api:app
  src/intent/
    __init__.py                  version
    config.py                    Settings.from_env(), resources_dir()
    errors.py                    IntentError + subclasses, Errors registry
    schemas.py                   TokenView LanguageCandidate LanguageInfo
                                 RuleView IntentMatch IntentResult ResourceReport
    normalize.py                 normalise(), Normalised, offset index map
    rules.py                     Rule RuleSet IntentSpec, load + validation
    language.py                  LanguageIdentifier, script + profiles
    matching.py                  Matcher, negation, scoring, rank
    pipeline.py                  spaCy load w/ fallback, IntentResult assembly
    chat.py                      Chatbot, strategies, sessions, PII redaction
    service.py                   IntentService use cases + ConversationStore
    api.py                       FastAPI routes, error envelope, static files
    cli.py                       Typer app
    rules/pt.json  rules/en.json 11 rule-driven intents each
  resources/language_profiles.json  function words, exclusive + shared
  web/index.html  web/styles.css
  web/js/api.js format.js intents.js chat.js app.js
  tests/
    conftest.py                  blank pipes, settings, detect_with factory
    fixtures/pt_cases.json       the five spec fixtures + per-intent samples
    test_config.py  test_errors.py  test_schemas.py
    test_normalisation.py  test_rules.py  test_language.py
    test_matching.py  test_negation.py  test_pipeline.py
    test_service.py  test_chat.py  test_api.py  test_cli.py  test_assets.py
  data/.gitkeep
  docs/adr/0001-rules-as-data.md
  docs/adr/0002-lexicon-free-scoring.md
  docs/adr/0003-abstain-on-unsupported-language.md
```

`aa/app.py` already exists as an **empty stub** from an abandoned project. Task 1
replaces it. There is nothing in it to read.

---

## Task 1: Skeleton, settings, errors, schemas

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`,
  `resources/language_profiles.json` (empty object),
  `src/intent/__init__.py`, `src/intent/config.py`, `src/intent/errors.py`,
  `src/intent/schemas.py`, `data/.gitkeep`
- Modify: `app.py` (empty stub → uvicorn entrypoint)
- Test: `tests/test_config.py`, `tests/test_errors.py`, `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing. Root task.
- Produces:
  - `intent.__version__ == "0.1.0"`.
  - `intent.config.Settings` — frozen dataclass with fields, in this order:
    `languages: tuple[str, ...]`, `db_path: str`, `min_text_chars: int`,
    `max_text_chars: int`, `negation_window: int`, `max_intents: int`,
    `min_confidence: float`, `min_script_share: float`,
    `min_language_share: float`, `min_language_margin: float`,
    `mixed_language_ratio: float`, `batch_max_items: int`, `log_level: str`.
    `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings`,
    defaulting to `os.environ` when `env` is `None`. Raises `ValueError` naming
    the variable when a value will not coerce.
  - `intent.config.resources_dir() -> pathlib.Path`.
  - `intent.errors.Errors` with class attributes `INPUT_INVALID`,
    `UNSUPPORTED_LANGUAGE`, `UNKNOWN_LANGUAGE`, `UNKNOWN_INTENT`,
    `INVALID_RULES`, `MODEL_UNAVAILABLE`, `STORAGE_ERROR`, `INTERNAL`.
  - `intent.errors.IntentError(Exception)` built as
    `IntentError(code: str, message: str, details: dict | None = None)`,
    exposing `.code`, `.message`, `.details`, `.retryable`. Subclasses:
    `InputError`, `UnsupportedLanguageError`, `UnknownLanguageError`,
    `UnknownIntentError`, `RulesError`, `ModelUnavailableError`, `StorageError`.
  - `intent.schemas` frozen dataclasses, field order as listed because tests
    construct several of them positionally:
    - `TokenView(text, lemma, pos, is_stop, is_sentence_start, matched, negated)`
    - `LanguageCandidate(language, share, markers)`
    - `LanguageInfo(language, supported, reason, script, offset_preserved,
      candidates, min_share, margin)`
    - `RuleView(rule_id, type, value, weight, char_start, char_end, negated,
      matched_text, offsets_exact)`
    - `IntentMatch(intent, label, score, confidence, priority, min_score,
      rule_hits)`
    - `IntentResult(text, normalised, intents, primary, multi_intent,
      confidence, tokens, language, explanation, rules_version, trace,
      resources, duration_ms)`
    - `ResourceReport(models, rules, limitations, warnings)` where
      `models[language] = {"model": str, "fallback": bool}` and
      `rules[language] = {"version": str, "entries": int, "sha256": str}`.
      `model` is the loaded model name, or `f"blank({language})"` when the
      blank-pipe fallback is in use. `limitations` and `warnings` are
      `tuple[str, ...]` and callers pass tuples — dataclasses do not enforce
      annotations, so a test passing a list would silently agree with an
      annotation it contradicts. Task 9 is the producer.

- [ ] **Step 1: Create the venv and install dependencies**

```powershell
cd "C:\Users\Aluno\Downloads\Nova pasta\aa"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "spacy>=3.8,<4" "fastapi>=0.115" "uvicorn[standard]>=0.30" "typer>=0.12" "pydantic>=2.8" pytest httpx python-multipart
.\.venv\Scripts\python.exe -m spacy download pt_core_news_sm
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

If a model download fails, continue. The blank-pipe fallback is Task 9's job and
the whole suite must pass without any model installed.

- [ ] **Step 2: Write `pyproject.toml`**

Setuptools, src layout, `package-data` for `intent/rules/*.json`. Dependencies
exactly as installed in Step 1 — no version drift between the two files.
`[project.optional-dependencies]` gets `pt = ["pt-core-news-sm>=3.8,<3.9"]`,
`en = ["en-core-web-sm>=3.8,<3.9"]`, and `dev = ["pytest>=8", "httpx>=0.27"]`.
`[project.scripts] intent = "intent.cli:main"`. `[tool.pytest.ini_options]` sets
`testpaths = ["tests"]`, `pythonpath = ["src"]`, `addopts = "-q"`.

`pythonpath = ["src"]` lets pytest import `intent` without an editable install,
so a broken install can never be the reason a test fails.

- [ ] **Step 3: Write the failing tests for settings and errors**

`tests/test_config.py`:

```python
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
```

`resources_dir()` is `Path(__file__).resolve().parents[2] / "resources"`, so from
`src/intent/config.py` it resolves to the repo's `resources/`. Create
`resources/language_profiles.json` as `{}` in this task so the assertion has
something to find; Task 3 fills it in.

`tests/test_errors.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_errors.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'intent'`

- [ ] **Step 5: Implement `config.py`, `errors.py`, `__init__.py`**

`Settings` is `@dataclass(frozen=True)`. `from_env` coerces each field through a
small typed helper and includes the variable name in every `ValueError`.

`errors.py` defines `Errors` with the eight codes plus
`RETRYABLE: frozenset[str] = frozenset({Errors.MODEL_UNAVAILABLE, Errors.STORAGE_ERROR})`.
`IntentError.__init__` sets `self.retryable = code in Errors.RETRYABLE`. Each
subclass passes its own code as a default so `InputError("x")` works with one
argument.

- [ ] **Step 6: Write the failing tests for schemas**

```python
import dataclasses
import pytest
from intent.schemas import (IntentMatch, IntentResult, LanguageCandidate,
                            LanguageInfo, ResourceReport, RuleView, TokenView)


def test_token_view_fields():
    tv = TokenView("cancelar", "cancelar", "VERB", False, False, True, False)
    assert (tv.text, tv.lemma, tv.pos) == ("cancelar", "cancelar", "VERB")
    assert tv.matched is True
    assert tv.negated is False


def test_token_view_has_no_leaked_spacy_attributes():
    names = {f.name for f in dataclasses.fields(TokenView)}
    assert not any(n.endswith("_") for n in names)
    assert "sentiment" not in names


def test_token_view_is_frozen():
    tv = TokenView("a", "a", "NOUN", False, True, False, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tv.text = "b"


def test_intent_match_carries_min_score():
    # the UI renders "score / min_score", so the field has to exist here
    m = IntentMatch("cancelar", "Cancelar assinatura", 3.5, 0.78, 90, 3.0, ())
    assert m.min_score == 3.0


def test_language_candidate_shape():
    c = LanguageCandidate("pt", 0.8, ("quero", "quero"))
    assert c.language == "pt" and c.markers == ("quero", "quero")


def test_language_info_rejection_carries_the_diagnostics():
    info = LanguageInfo(None, False, "ambiguous_language", "Latin", True,
                        (LanguageCandidate("pt", 0.5, ()),
                         LanguageCandidate("en", 0.48, ())), 0.55, 0.02)
    assert info.margin == 0.02
    assert len(info.candidates) == 2


def test_resource_report_shape():
    r = ResourceReport({"pt": {"model": "blank(pt)", "fallback": True}},
                       {"pt": {"version": "1.0.0", "entries": 24, "sha256": "ab"}},
                       (), ())
    assert r.models["pt"]["fallback"] is True
    assert r.rules["pt"]["entries"] == 24


def test_rule_view_has_the_documented_fields():
    names = {f.name for f in dataclasses.fields(RuleView)}
    assert names == {"rule_id", "type", "value", "weight", "char_start",
                     "char_end", "negated", "matched_text", "offsets_exact"}


def test_intent_result_field_order():
    assert [f.name for f in dataclasses.fields(IntentResult)] == [
        "text", "normalised", "intents", "primary", "multi_intent", "confidence",
        "tokens", "language", "explanation", "rules_version", "trace",
        "resources", "duration_ms"]
```

- [ ] **Step 7: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py -v`
Expected: `ImportError: cannot import name 'TokenView'`

- [ ] **Step 8: Implement `schemas.py`**

Frozen dataclasses exactly per the Interfaces block, in the listed field order.

- [ ] **Step 9: Replace the empty `app.py` stub**

```python
import uvicorn

from intent.api import create_app

app = create_app()


def main() -> None:
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
```

`intent.api` arrives in Task 12, so this file is not importable until then.
Nothing imports it before then.

- [ ] **Step 10: Write `.gitignore`, `.env.example`, `Makefile`, `data/.gitkeep`**

`.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `data/*`, `!data/.gitkeep`,
`.pytest_cache/`, `.env`.

`.env.example`: all thirteen `INTENT_*` variables with their defaults as
comments, nothing else.

`Makefile`, tab-indented:

```makefile
PY := .venv/Scripts/python.exe

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m spacy download pt_core_news_sm
	$(PY) -m spacy download en_core_web_sm

test:
	$(PY) -m pytest

serve:
	$(PY) app.py
```

- [ ] **Step 11: Run the suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: `test_config.py`, `test_errors.py`, `test_schemas.py` all pass.

- [ ] **Step 12: Commit** (conditional on git)

```bash
git add pyproject.toml Makefile .gitignore .env.example app.py data/.gitkeep resources src/intent tests
git commit -m "feat: project skeleton, settings, error codes, value objects"
```

---

## Task 2: Normalisation and the offset index map

The subtlest invariant in the project (§5.1). Its own task and its own reviewer,
because every offset the UI highlights depends on it and a silent off-by-N is
invisible in the happy path.

**Files:**
- Create: `src/intent/normalize.py`, `tests/test_normalisation.py`
- Test: `tests/test_normalisation.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Normalised(text: str, index_map: tuple[int, ...])` — frozen.
    `index_map[i]` is the index in the **original** string that produced
    `text[i]`. `len(index_map) == len(text)` always.
  - `normalise(text: str) -> Normalised` — lowercase, NFD, drop combining marks,
    collapse whitespace runs to one space, strip leading and trailing space.
  - `to_original_offset(n: Normalised, offset: int) -> int` — translate a
    normalised index to an original index, clamped to `[0, len(original)]`.
  - `strip_accents(text: str) -> str` — NFD plus dropping
    `unicodedata.combining` characters. No lowercasing. Used by rule loading.
  - `has_cased_character(text: str) -> bool` — True when some character is
    alphabetic and not a combining mark. Drives the `empty_text` gate.

- [ ] **Step 1: Write the failing tests**

```python
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
    # the whole point: "nao" is one character shorter than "não", so a naive
    # offset for "quero" would be off by one
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


def test_offset_clamps_instead_of_raising():
    n = normalise("oi")
    assert to_original_offset(n, 0) == 0
    assert to_original_offset(n, 99) == 2


def test_strip_accents_does_not_lowercase():
    assert strip_accents("Não") == "Nao"


def test_strip_accents_leaves_other_scripts_alone():
    assert strip_accents("취소해줘") == "취소해줘"


def test_empty_and_blank_inputs():
    assert normalise("").text == ""
    assert normalise("   \n\t ").text == ""


def test_newlines_collapse_to_a_single_space():
    assert normalise("não\nquero\tcancelar").text == "nao quero cancelar"


def test_has_cased_character():
    assert has_cased_character("abc") is True
    assert has_cased_character("não") is True
    assert has_cased_character("   ") is False
    assert has_cased_character("!!!") is False
    assert has_cased_character("12345") is False
    assert has_cased_character("🙂") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_normalisation.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.normalize'`

- [ ] **Step 3: Implement `normalize.py`**

Build `text` and `index_map` in a single pass over the original string so the two
cannot drift. For each original index: decompose; if it expands, emit the base
character once and record the base character's original index for every character
the expansion emits; if it is whitespace, emit a single `" "` for the whole run
carrying the run's first index; otherwise emit the lowercased character with its
own index. `.strip()` at the end must trim `text` and `index_map` together, not
independently.

`to_original_offset` returns `index_map[offset]` when in range, else clamps to
the nearest end. Deciding what an unmappable boundary means is the pipeline's
job, not this function's; the pipeline sets `offsets_exact = False` there.

- [ ] **Step 4: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_normalisation.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit** (conditional)

```bash
git add src/intent/normalize.py tests/test_normalisation.py
git commit -m "feat: normalisation with an offset index map"
```

---

## Task 3: Language identification

**Files:**
- Create: `src/intent/language.py`, `resources/language_profiles.json` (fill in)
- Test: `tests/test_language.py`

**Interfaces:**
- Consumes: `intent.normalize.normalise`, `has_cased_character`; `Settings`.
- Produces:
  - `intent.language.Reason` with `OK = "ok"`,
    `INSUFFICIENT_SIGNAL = "insufficient_signal"`,
    `AMBIGUOUS_LANGUAGE = "ambiguous_language"`,
    `UNSUPPORTED_SCRIPT = "unsupported_script"`, `EMPTY_TEXT = "empty_text"`.
  - `load_profiles(path: pathlib.Path | None = None) -> dict` reading
    `{"languages": {"pt": {"exclusive": [...], "shared": [...]}, "en": {...}}}`.
  - `LanguageIdentifier(profiles: dict, settings: Settings)` with
    `identify(text: str) -> LanguageInfo`. `min_share` and `margin` are `None`
    when the decision is accepted.

- [ ] **Step 1: Write the failing tests**

```python
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
    # seven tokens, not five: with "the quer quero the my" the shares are forced
    # to 3/5 and 2/5 — every implementation of the step-3 algorithm counts
    # exclusive-marker occurrences over the word tokens — giving a margin of
    # 0.2, which is above min_language_margin=0.15 and so reports `ok`/`en`.
    # Growing the vocabulary cannot change that, because every token in the
    # string is already recognised. Seven tokens gives 4/7 and 3/7, a margin of
    # 0.1428, which trips it while both shares stay above min_language_share and
    # mixed_language_ratio.
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
    assert info.min_share is not None


def test_script_is_reported(identifier):
    assert identifier.identify("quero cancelar").script == "Latin"
    assert identifier.identify("취소해줘").script == "CJK"


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_language.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.language'`

- [ ] **Step 3: Write `resources/language_profiles.json`**

`pt.exclusive` at least: `o a os as de do da dos das que e em para um uma com
nao sim por mais como ao aos pelo sob entre depois antes quero preciso posso qual
quando muito tambem ja na muito obrigado`. `pt.shared`: `plano conta assinatura
cancelamento erro problema preco valor`.

`en.exclusive` at least: `i the a an of to and is are was were for in on at my me
want need can how what when where which not do does please`. `en.shared`: `plan
account subscription cancel error problem price value`.

Keep the two `shared` lists identical in content — shared markers are excluded
from scoring entirely, so their entries are documentation.

- [ ] **Step 4: Implement `language.py`**

`identify`, in order:

1. `n = normalise(text)`. If not `has_cased_character(n.text)` → `LanguageInfo`
   with `supported=False`, `reason=EMPTY_TEXT`, `language=None`, `script=None`,
   `offset_preserved=True`, `candidates=()`.
2. Script shares over cased characters, classified by the first word of
   `unicodedata.name(ch, "")`: `LATIN`→`Latin`; `CJK`/`HIRAGANA`/`KATAKANA`/
   `HANGUL`→`CJK`; `CYRILLIC`→`Cyrillic`; `ARABIC`→`Arabic`. If the Latin share
   is below `settings.min_script_share` → `reason=UNSUPPORTED_SCRIPT`.
3. Exclusive-marker shares per configured language over the word tokens. Drop
   any language below `settings.mixed_language_ratio`. Sort by share descending,
   then language ascending.
4. No candidates, or top below `settings.min_language_share` →
   `INSUFFICIENT_SIGNAL`. Gap to second below `settings.min_language_margin` →
   `AMBIGUOUS_LANGUAGE`. Otherwise the top language with `reason=OK`.

Populate `candidates` with the real `LanguageCandidate` list in every branch
after step 2, including the rejected ones, and set `min_share`/`margin` whenever
the decision is not `OK`.

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_language.py -v`
Expected: 18 passed. If `test_genuinely_mixed_is_ambiguous` fails, the
`exclusive` lists are too small for the margin to trip — add markers rather than
lowering `min_language_margin`. If it fails because the margin is *too large*,
lengthen the test input so the shares fall closer together; never lower
`min_language_margin`, which is a spec-fixed default.

- [ ] **Step 6: Commit** (conditional)

```bash
git add src/intent/language.py resources/language_profiles.json tests/test_language.py
git commit -m "feat: two-stage language identification with abstention"
```

---

## Task 4: Rule loading and validation

**Files:**
- Create: `src/intent/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `intent.errors.RulesError`, `UnknownIntentError`,
  `UnknownLanguageError`; `intent.normalize.strip_accents`.
- Produces:
  - `RuleType` with `KEYWORD = "keyword"`, `LEMMA = "lemma"`,
    `PATTERN = "pattern"`, `ALL = "all"`.
  - `Rule(rule_id, type, value: str | tuple[str, ...], weight: float,
    compiled: re.Pattern | None)` — frozen. `compiled` is non-`None` only for
    `PATTERN`.
  - `IntentSpec(id, label, priority, min_score, rules, examples)` — frozen.
  - `RuleSet(language, version, intents, negators, negation_boundaries, sha256)`
    — frozen, with `by_id(id) -> IntentSpec` raising `UnknownIntentError`.
  - `validate_document(document: dict) -> None` — raises `RulesError` with the
    JSON path in `.details["path"]`.
  - `load_ruleset(language: str, path: pathlib.Path | None = None) -> RuleSet`.
  - `load_all_rulesets(languages: tuple[str, ...]) -> dict[str, RuleSet]`.

`rule_id` is `f"{intent_id}:{index}"` — `cancelar:0`, `cancelar:1`. It is the
dedupe key in Task 5 and the trace key everywhere.

- [ ] **Step 1: Write the failing tests**

```python
import json
import pathlib
import pytest
from intent.errors import RulesError, UnknownIntentError, UnknownLanguageError
from intent.rules import load_all_rulesets, load_ruleset, validate_document

VALID = {
    "language": "pt", "version": "1.0.0",
    "intents": [{
        "id": "cancelar", "label": "Cancelar assinatura", "priority": 90,
        "min_score": 3.0,
        "rules": [{"type": "keyword", "value": "cancelar", "weight": 2.0}],
        "examples": ["quero cancelar"],
    }],
    "negators": ["não", "deixar de"],
    "negation_boundaries": ["mas", "porque", "e"],
}


def _write(document, tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _load(document, tmp_path):
    return load_ruleset("pt", path=_write(document, tmp_path))


def test_valid_document_passes():
    validate_document(VALID)


@pytest.mark.parametrize("mutate,path", [
    (lambda d: d["intents"][0]["rules"].append({"type": "nope", "value": "x", "weight": 1.0}),
     "intents.0.rules.1.type"),
    (lambda d: d["intents"][0]["rules"][0].update(value="duas palavras"),
     "intents.0.rules.0.value"),
    (lambda d: d["intents"][0]["rules"][0].update(value="[unclosed"),
     "intents.0.rules.0.value"),
    (lambda d: d["intents"][0]["rules"][0].update(weight=0),
     "intents.0.rules.0.weight"),
    (lambda d: d["intents"][0]["rules"][0].update(weight="two"),
     "intents.0.rules.0.weight"),
    (lambda d: d["intents"][0]["rules"][0].update(weight=float("inf")),
     "intents.0.rules.0.weight"),
    (lambda d: d["intents"][0].update(min_score=0), "intents.0.min_score"),
    (lambda d: d["intents"][0].update(id="Cancelar"), "intents.0.id"),
    (lambda d: d["intents"][0].update(priority="alta"), "intents.0.priority"),
    (lambda d: d["intents"].append(dict(d["intents"][0])), "intents.1.id"),
    (lambda d: d.update(version="1.0"), "version"),
    (lambda d: d["intents"][0]["rules"].append(
        {"type": "all", "value": ["so"], "weight": 1.0}), "intents.0.rules.1.value"),
    (lambda d: d["intents"][0]["rules"].append(
        {"type": "all", "value": ["a", "b c"], "weight": 1.0}), "intents.0.rules.1.value"),
    (lambda d: d["intents"][0]["rules"].append(
        dict(d["intents"][0]["rules"][0])), "intents.0.rules.1"),
    (lambda d: d.update(intents=[]), "intents"),
])
def test_invalid_documents_are_rejected_with_a_path(mutate, path):
    document = json.loads(json.dumps(VALID))
    mutate(document)
    with pytest.raises(RulesError) as exc:
        validate_document(document)
    assert exc.value.details["path"].startswith(path), exc.value.details


def test_nan_weight_is_rejected():
    # nan <= 0 is False, so a naive positivity check lets it through and it
    # then poisons every ranking
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"][0]["weight"] = float("nan")
    with pytest.raises(RulesError):
        validate_document(document)


def test_integer_weight_is_accepted(tmp_path):
    # Review Focus #5: these files are hand-edited by non-Python speakers
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"][0]["weight"] = 2
    validate_document(document)
    rule = _load(document, tmp_path).intents[0].rules[0]
    assert rule.weight == 2.0
    assert isinstance(rule.weight, float)


def test_duplicate_type_and_value_within_one_intent_is_rejected():
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"].append(
        {"type": "keyword", "value": "cancelar", "weight": 1.0})
    with pytest.raises(RulesError):
        validate_document(document)


def test_the_same_value_in_two_different_intents_is_fine(tmp_path):
    document = json.loads(json.dumps(VALID))
    other = dict(document["intents"][0])
    other["id"] = "comprar"
    document["intents"].append(other)
    validate_document(document)
    assert len(_load(document, tmp_path).intents) == 2


def _with_pattern(value, tmp_path):
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"] = [{"type": "pattern", "value": value, "weight": 3.0}]
    return _load(document, tmp_path)


def test_pattern_is_compiled_with_implicit_word_boundaries(tmp_path):
    rule = _with_pattern("assinar", tmp_path).intents[0].rules[0]
    assert rule.compiled.search("vou assinar o plano") is not None
    # the point: it must not fire inside a longer word
    assert rule.compiled.search("vou desassinar o plano") is None


def test_pattern_value_is_accent_stripped_at_load(tmp_path):
    rule = _with_pattern(r"não\s+funciona", tmp_path).intents[0].rules[0]
    assert rule.compiled.search("nao funciona") is not None


def test_rule_ids_are_indexed_by_position(tmp_path):
    document = json.loads(json.dumps(VALID))
    document["intents"][0]["rules"] = [
        {"type": "keyword", "value": "cancelar", "weight": 2.0},
        {"type": "lemma", "value": "cancelamento", "weight": 2.0},
    ]
    ruleset = _load(document, tmp_path)
    assert [r.rule_id for r in ruleset.intents[0].rules] == ["cancelar:0", "cancelar:1"]


def test_by_id_raises_for_an_unknown_intent(tmp_path):
    with pytest.raises(UnknownIntentError):
        _load(VALID, tmp_path).by_id("nao_existe")


def test_by_id_returns_the_spec(tmp_path):
    ruleset = _load(VALID, tmp_path)
    assert ruleset.by_id("cancelar").priority == 90


def test_missing_file_raises_unknown_language(tmp_path):
    with pytest.raises(UnknownLanguageError):
        load_ruleset("pt", path=tmp_path / "missing.json")


def test_default_path_is_in_package_and_names_itself():
    # src/intent/rules/pt.json does not exist until Task 8, so this pins the
    # resolution without needing the file: the error must name the path that
    # would have been read.
    with pytest.raises(UnknownLanguageError) as exc:
        load_ruleset("pt")
    named = str(exc.value)
    assert "rules" in named and named.rstrip().endswith("pt.json")
    assert "resources" not in named


def test_load_all_rulesets_reports_the_language_it_tried():
    with pytest.raises(UnknownLanguageError) as exc:
        load_all_rulesets(("pt",))
    assert "pt" in str(exc.value)


def test_sha256_is_stable_and_content_dependent(tmp_path):
    a = _load(VALID, tmp_path)
    b = _load(json.loads(json.dumps(VALID)), tmp_path)
    c = _with_pattern("x", tmp_path)
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


def test_negators_and_boundaries_are_loaded(tmp_path):
    ruleset = _load(VALID, tmp_path)
    assert "deixar de" in ruleset.negators
    assert ruleset.negation_boundaries == frozenset({"mas", "porque", "e"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rules.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.rules'`

- [ ] **Step 3: Implement `rules.py`**

`validate_document` checks in this order, so the reported path is the most
specific one available: `language` present and two letters; `version` matching
`^\d+\.\d+\.\d+$`; `intents` a non-empty list; per intent `id` matching
`^[a-z][a-z0-9_]*$` and unique, `label` non-empty, `priority` an `int`,
`min_score` > 0, `rules` a non-empty list, `examples` a list of strings. Per
rule: `type` in `RuleType`; for `keyword`/`lemma` a string value with no
whitespace; for `pattern` a string that compiles both bare and wrapped in
`(?<!\w)…(?!\w)`; for `all` a list of at least 2 strings each without
whitespace. `weight` via `float()` inside `try`, rejecting `nan`, `inf` and
values ≤ 0. Duplicate `(type, value)` within one intent rejected.

`pattern` values are `strip_accents`ed, then compiled as
`re.compile(r"(?<!\w)" + value + r"(?!\w)")`. A pattern written with accents
therefore still matches an unaccented message.

`load_ruleset` computes `sha256` over the raw file bytes, so it changes on any
content change including whitespace.

When `path` is `None` the default is **in-package**, not under `resources/`:
`Path(__file__).resolve().parent / "rules" / f"{language}.json"`. That is what
`package-data` for `intent/rules/*.json` ships, so the same code works from a
source checkout and from an installed wheel. `resources_dir()` stays the
location of `language_profiles.json` only — the two roots are deliberately
different and nothing should look for rules under `resources/`.

`load_all_rulesets` maps each language through that same default and raises
`UnknownLanguageError` naming the resolved path when the file is absent.

- [ ] **Step 4: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rules.py -v`
Expected: all passed

- [ ] **Step 5: Commit** (conditional)

```bash
git add src/intent/rules.py tests/test_rules.py
git commit -m "feat: declarative rule loading with strict validation"
```

---

## Task 5: Rule evaluation and scoring

The heart of the engine: four rule types, `all` token consumption, per-rule
dedupe, substring safety. Negation is deliberately not here — Task 6.

**Files:**
- Create: `src/intent/matching.py`, `tests/test_matching.py`
- Modify: `tests/conftest.py` (add the shared fixtures)
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: `intent.rules.Rule`, `RuleSet`, `RuleType`;
  `intent.normalize.Normalised`, `to_original_offset`; a spaCy `Doc`.
- Produces:
  - `RawHit(rule_id, type, value, weight, char_start, char_end, matched_text,
    offsets_exact, token_indices)` — **mutable**, because Task 6 flips
    `negated` on it. Add `negated: bool = False` as the last field.
  - `IntentScore(intent_id, label, score, priority, min_score, hits)` —
    `hits: tuple[RawHit, ...]`.
  - `Matcher(ruleset: RuleSet, settings: Settings)` with
    `evaluate(doc: Doc, normalised: Normalised) -> list[IntentScore]`, returning
    an entry for every intent that had at least one hit, including those below
    `min_score`. Filtering is Task 7's job, which is what lets these tests
    assert on raw scores.

- [ ] **Step 1: Add the shared fixtures to `tests/conftest.py`**

```python
import json
import pathlib

import pytest
import spacy

from intent.config import Settings
from intent.matching import Matcher
from intent.normalize import normalise
from intent.rules import load_ruleset


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
```

`negators` and `boundaries` are named parameters, not part of `**env_overrides`.
Forwarding them to `Settings.from_env` would raise `ValueError`, since no
`INTENT_NEGATORS` setting exists.

- [ ] **Step 2: Write the failing tests**

```python
import pytest
import spacy


KEYWORD_CANCELAR = [{"type": "keyword", "value": "cancelar", "weight": 2.0}]


def test_keyword_matches_an_exact_token(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("quero cancelar o plano")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].token_indices == (1,)


def test_keyword_does_not_match_inside_a_longer_word(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("quero descancelar o plano") == []


def test_pattern_matches_across_punctuation(detect_with, blank_nlp):
    rules = [{"type": "pattern", "value": r"cancelar\s+(meu\s+)?plano", "weight": 3.0}]
    scores = detect_with(blank_nlp, rules)("quero cancelar, meu plano, hoje")
    assert scores[0].score == 3.0


def test_pattern_hits_map_to_a_token_span(detect_with, blank_nlp):
    rules = [{"type": "pattern", "value": r"nao\s+funciona", "weight": 2.5}]
    scores = detect_with(blank_nlp, rules)("o app nao funciona desde ontem")
    assert scores[0].hits[0].token_indices


def test_pattern_does_not_match_inside_a_longer_word(detect_with, blank_nlp):
    # Review Focus #2
    rules = [{"type": "pattern", "value": "assinar", "weight": 2.0}]
    assert detect_with(blank_nlp, rules)("vou desassinar o plano") == []


def test_all_requires_every_string(detect_with, blank_nlp):
    rules = [{"type": "all", "value": ["cancelar", "assinatura"], "weight": 3.5}]
    assert detect_with(blank_nlp, rules)("quero cancelar") == []
    assert detect_with(blank_nlp, rules)("quero cancelar a assinatura")[0].score == 3.5


def test_all_tokens_are_consumed_and_not_double_counted(detect_with, blank_nlp):
    # without consumption, one mention of "cancelar" would score 3.5 + 2.0
    rules = [{"type": "all", "value": ["cancelar", "assinatura"], "weight": 3.5},
             {"type": "keyword", "value": "cancelar", "weight": 2.0}]
    assert detect_with(blank_nlp, rules)("quero cancelar a assinatura")[0].score == 3.5


def test_consumption_does_not_depend_on_rule_order(detect_with, blank_nlp):
    # the same two rules in the opposite order must score identically, or the
    # result would depend on how the JSON happened to be written
    keyword_first = [{"type": "keyword", "value": "cancelar", "weight": 2.0},
                     {"type": "all", "value": ["cancelar", "assinatura"],
                      "weight": 3.5}]
    all_first = list(reversed(keyword_first))
    assert (detect_with(blank_nlp, keyword_first)("quero cancelar a assinatura")[0].score
            == detect_with(blank_nlp, all_first)("quero cancelar a assinatura")[0].score
            == 3.5)


def test_repetition_does_not_inflate_the_score(detect_with, blank_nlp):
    # Review Focus #1
    once = detect_with(blank_nlp, KEYWORD_CANCELAR)("quero cancelar")
    thrice = detect_with(blank_nlp, KEYWORD_CANCELAR)(
        "cancelar cancelar cancelar cancelar")
    assert once[0].score == 2.0
    assert thrice[0].score == 2.0
    # every occurrence is still visible in the trace
    assert len(thrice[0].hits) == 4


def test_an_intent_with_no_hits_is_absent(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("quero um café") == []


def test_hit_offsets_point_at_the_original_accented_text(detect_with, blank_nlp):
    rules = [{"type": "pattern", "value": r"nao\s+quero", "weight": 2.0}]
    text = "olá, não quero isso"
    hit = detect_with(blank_nlp, rules)(text)[0].hits[0]
    # slicing the ORIGINAL and expecting the accent is the only proof the
    # index map is actually being used
    assert text[hit.char_start:hit.char_end].lower() == "não quero"
    assert hit.offsets_exact is True


def test_lemma_rule_when_the_model_can_lemmatise(detect_with):
    try:
        nlp = spacy.load("pt_core_news_sm")
    except OSError:
        pytest.skip("pt_core_news_sm not installed")
    rules = [{"type": "lemma", "value": "cancelamento", "weight": 2.0}]
    assert detect_with(nlp, rules)("quero fazer um cancelamento")[0].score == 2.0


def test_lemma_rule_is_skipped_silently_without_a_lemmatiser(detect_with, blank_nlp):
    # a blank pipe has no lemmatiser; the rule is dropped, not an error
    rules = [{"type": "lemma", "value": "cancelamento", "weight": 2.0}]
    assert detect_with(blank_nlp, rules)("quero fazer um cancelamento") == []
```

- [ ] **Step 3: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_matching.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.matching'`

- [ ] **Step 4: Implement `matching.py`**

`evaluate` walks `ruleset.intents` and, for each intent, runs **two passes** so
the result never depends on the order the rules happen to sit in the JSON:

**Pass 1 — `ALL` rules only.** For each, gather token indices per string; if any
is empty there is no hit. Otherwise the **union** of every string's token indices
goes into `consumed`, while each emitted `RawHit` carries **only its own
string's** token span as `token_indices` — not the union. Task 6's leftward
negation scan reads `min(hit.token_indices)`, so a per-string span lets each
occurrence be negated independently; a union would hand every hit the same start
token and negate the whole rule from one occurrence. The weight is credited once
per rule, not once per emitted hit.

**Pass 2 — every other rule**, skipping any token already in `consumed`:

- `KEYWORD`: tokens where `token.lower_` equals `value.lower()` or
  `strip_accents(value)`.
- `LEMMA`: the same against `token.lemma_.lower()`; skipped entirely when
  `not doc.has_annotation("LEMMA")`.
- `PATTERN`: `rule.compiled.finditer(normalised.text)`, ends translated through
  `to_original_offset`. `token_indices` is every token satisfying
  `token.idx < char_end and token.idx + len(token.text) > char_start`. When the
  regex starts mid-token, clamp the span to the token and set
  `offsets_exact = False`.

Doing `ALL` first is what makes consumption meaningful. A single pass would let
`keyword cancelar` score 2.0 before `all ["cancelar","assinatura"]` claimed the
same token, and the total would depend on array order — a rules file is content,
and content order should not change a score.

A rule contributes its `weight` once even when it matched many tokens, but emits
one `RawHit` per token span so the trace shows every occurrence.

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_matching.py -v`
Expected: all passed, with at most one skip when the model is absent

- [ ] **Step 6: Commit** (conditional)

```bash
git add src/intent/matching.py tests/conftest.py tests/test_matching.py
git commit -m "feat: four rule types, token consumption, per-rule scoring"
```

---

## Task 6: Negation

Its own task because it is the most valuable diagnostic in the UI and the easiest
thing to get subtly wrong (§5.2).

**Files:**
- Create: `tests/test_negation.py`
- Modify: `src/intent/matching.py` (add the negation pass)

**Interfaces:**
- Consumes: `Matcher` from Task 5; `RuleSet.negators`,
  `RuleSet.negation_boundaries`; `Settings.negation_window`.
- Produces: `Matcher.evaluate` now sets `hit.negated` and excludes negated hits
  from the intent's `score` while keeping them in `hits`. Each rule contributes
  weight once, so a negated rule removes its whole weight from the intent.

- [ ] **Step 1: Write the failing tests**

```python
KEYWORD_CANCELAR = [{"type": "keyword", "value": "cancelar", "weight": 2.0}]
KEYWORD_COMPRAR = [{"type": "keyword", "value": "comprar", "weight": 2.0}]


def test_a_negator_before_the_match_negates(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não quero cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True


def test_negator_immediately_before_the_match(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("não cancelar")[0].score == 0.0


def test_a_negator_beyond_the_window_does_not_negate(detect_with, blank_nlp):
    # "não" is four tokens back and the window is three
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não quero mesmo assim cancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_negation_does_not_cross_a_clause_boundary(detect_with, blank_nlp):
    # "não gostei, mas quero cancelar" cannot fail this: the negator is 5 tokens
    # left of the hit, so the 3-token window never reaches it and the test passes
    # with the whole boundary branch deleted. The negator must sit *inside* the
    # window with the boundary between it and the hit.
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)("não, mas vou cancelar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_newline_is_a_clause_boundary(detect_with, blank_nlp):
    # Same trap as above: "não cancelar\n..." puts the negator out of reach.
    scores = detect_with(blank_nlp, KEYWORD_COMPRAR)("não\nquero comprar")
    assert scores[0].score == 2.0
    assert scores[0].hits[0].negated is False


def test_multi_word_negator(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("deixar de cancelar")[0].score == 0.0


def test_a_message_that_is_only_a_negator_does_not_error(detect_with, blank_nlp):
    assert detect_with(blank_nlp, KEYWORD_CANCELAR)("não") == []


def test_negation_is_per_occurrence_not_per_intent(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR)(
        "não quero cancelar, mas vou cancelar depois")
    assert scores[0].score == 2.0
    assert sum(1 for h in scores[0].hits if h.negated) == 1


def test_a_negated_rule_removes_its_whole_weight(detect_with, blank_nlp):
    rules = KEYWORD_CANCELAR + [
      {"type": "pattern", "value": r"quero\s+cancelar", "weight": 3.0}]
    scores = detect_with(blank_nlp, rules)("não quero cancelar")
    assert scores[0].score == 0.0
    assert all(h.negated for h in scores[0].hits)

def test_a_rule_that_spans_its_own_negator_is_not_negated(detect_with, blank_nlp):
    # `suporte`'s `pattern nao\s+(funciona|abre|carrega|responde)` exists to match
    # the negative statement *as* the signal. Its span starts on the negator, so the
    # leftward scan has nothing to find and it must survive intact. Written here
    # because it is the inverse of the test above and the pair pins the rule.
    # `detect_with` takes a flat list of rules, not intent objects.
    rules = [{"type": "pattern", "value": r"nao\s+(funciona|abre)", "weight": 2.5}]
    scores = detect_with(blank_nlp, rules)("o app não funciona")
    assert scores[0].score == 2.5
    assert all(h.negated is False for h in scores[0].hits)


def test_negators_are_configurable(detect_with, blank_nlp):
    scores = detect_with(blank_nlp, KEYWORD_CANCELAR,
                         negators=["sem querer"])("sem querer cancelar")
    assert scores[0].score == 0.0
    assert scores[0].hits[0].negated is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_negation.py -v`
Expected: failures — negation is not detected yet, so
`test_a_negator_before_the_match_negates` sees `score == 2.0`

- [ ] **Step 3: Implement the negation pass in `matching.py`**

After collecting hits for an intent, for each hit take its first token index
(`min(hit.token_indices)`, or `None` → not negated) and walk backwards up to
`settings.negation_window` tokens. At each position, **check the negator first,
then the boundary**:

- A hit whose own span starts on a negator is **not** self-negated by that negator:
  the rule author wrote the negative into the pattern (`suporte`'s `nao\s+(funciona
  |abre|carrega|responde)`, `reclamacao`'s `nao aceito`), and the scan starts at
  `min(token_indices) - 1`, i.e. before it, so it never sees it. Do not add a
  "contains a negator" guard — it would be dead code that reads as if it mattered.
  The behaviour is pinned by `test_a_rule_that_spans_its_own_negator_is_not_negated`.
  Note this is not an unconditional exemption: a *different* negator within
  `negation_window` still negates the hit ("não quero não funciona" negates the
  second occurrence), which is correct — the outer negator outranks the pattern.
- Negators are phrases, matched on the normalised token text. Build a lookup
  grouping negators by token length. At position `p`, for `k` from the longest
  negator length down to 1, test whether the `k` tokens ending at `p` equal that
  negator; on a match, mark negated and stop the scan.
- **Task 4 stores `negators` and `negation_boundaries` raw** — it does not strip
  accents or lowercase them — while Task 8's catalogue ships them accented
  (`não`, `porém`, `então`). Comparing raw catalogue text against accent-stripped
  tokens would never match, and every negation test would fail at once. So
  `matching.py` owns the folding: `strip_accents` both the catalogue phrase and
  the token text before comparing. Do not "fix" this by rewriting the catalogue
  in ASCII — `não` is the correct spelling for the human editing the file, and
  rule *values* are accent-free only because they are compiled into regexes and
  compared against normalised text. Negators are prose, not patterns.
- Then, if the token is `.` `,` `;` `:` `!` `?` or a newline, stop without
  negating.
- Then, if the token's normalised text is in `ruleset.negation_boundaries`, stop
  without negating.

Punctuation and newline handling is hardcoded in `matching.py` and documented
there as such. A content file must not be able to disable punctuation handling,
because a rules author who omits a comma from `negation_boundaries` would
silently reintroduce cross-clause negation.

- [ ] **Step 4: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_negation.py tests/test_matching.py -v`
Expected: all passed. `tests/test_negation.py` also keeps the two original
out-of-window boundary messages and adds three in-window replacements, so the file
is strictly stronger than the two tests below; the originals are kept as
documentation of the trap and are marked as such in comments.
Task 5's tests must still pass — most of those messages do
contain a negator (`test_matching.py:86`), so this is a real check that the
negation pass did not change un-negated behaviour, not a formality.

- [ ] **Step 5: Commit** (conditional)

```bash
git add src/intent/matching.py tests/test_negation.py
git commit -m "feat: clause-bounded negation window with per-hit traces"
```

---

## Task 7: Ranking, multi-intent, and the confidence gate

**Files:**
- Modify: `src/intent/matching.py`
- Test: `tests/test_pipeline.py` (created here; Task 9 extends it)

**Interfaces:**
- Consumes: `IntentScore` from Task 5; `Settings`.
- Produces, in `intent.matching`:
  - `UNKNOWN_INTENT = "desconhecido"`.
  - `confidence_for(score: float) -> float` — `score / (score + 1)`, rounded
    to 4.
  - `RankedResult(kept, dropped, primary, multi_intent, confidence, reason)` —
    frozen. `kept: tuple[IntentMatch, ...]`, `dropped: tuple[IntentScore, ...]`,
    `reason` is `"ok"` or `"low_confidence"`.
  - `rank(scores: list[IntentScore], settings: Settings) -> RankedResult`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from intent.matching import (UNKNOWN_INTENT, IntentScore, confidence_for, rank)


def _score(intent_id, score, priority, min_score=1.0):
    return IntentScore(intent_id, intent_id, score, priority, min_score, ())


def test_confidence_curve():
    assert confidence_for(0.0) == 0.0
    assert confidence_for(1.0) == 0.5
    assert confidence_for(3.0) == 0.75


def test_below_min_score_is_dropped(settings):
    r = rank([_score("comprar", 1.5, 60, min_score=2.0)], settings)
    assert r.kept == ()
    assert r.primary == UNKNOWN_INTENT
    assert r.dropped[0].intent_id == "comprar"


def test_a_tie_is_broken_by_priority(settings):
    r = rank([_score("suporte", 3.0, 50), _score("cancelar", 3.0, 90)], settings)
    assert r.primary == "cancelar"


def test_equal_priority_is_broken_by_id_deterministically(settings):
    forward = rank([_score("suporte", 3.0, 50), _score("acesso", 3.0, 50)], settings)
    reverse = rank([_score("acesso", 3.0, 50), _score("suporte", 3.0, 50)], settings)
    assert forward.primary == "acesso"
    assert reverse.primary == forward.primary


def test_multi_intent_is_reported_and_capped(settings):
    scores = [_score("cancelar", 5.0, 90), _score("comprar", 4.0, 60),
              _score("suporte", 3.0, 50), _score("duvida", 2.5, 40)]
    r = rank(scores, settings)
    assert r.multi_intent is True
    assert len(r.kept) == 3          # INTENT_MAX_INTENTS
    assert [m.intent for m in r.kept] == ["cancelar", "comprar", "suporte"]


def test_a_single_intent_is_not_multi(settings):
    assert rank([_score("cancelar", 5.0, 90)], settings).multi_intent is False


def test_the_confidence_gate_produces_desconhecido_but_keeps_the_evidence(settings):
    r = rank([_score("duvida", 0.4, 40, min_score=0.1)], settings)
    assert r.primary == UNKNOWN_INTENT
    assert r.kept == ()
    assert r.reason == "low_confidence"
    assert r.dropped          # the near-miss stays visible to the UI


def test_no_input_produces_desconhecido(settings):
    r = rank([], settings)
    assert r.primary == UNKNOWN_INTENT
    assert r.confidence == 0.0
    assert r.multi_intent is False
    assert r.reason == "ok"


def test_kept_matches_carry_min_score(settings):
    r = rank([_score("cancelar", 4.0, 90, min_score=3.0)], settings)
    assert r.kept[0].min_score == 3.0
    assert r.kept[0].confidence == round(4.0 / 5.0, 4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: `ImportError: cannot import name 'rank'`

- [ ] **Step 3: Implement `rank` in `matching.py`**

Filter to `score >= min_score`, sort by `(-score, -priority, intent_id)`, cap
at `settings.max_intents`. Apply `confidence_for` to the top entry; if it is
below `settings.min_confidence`, return `kept=()`, `reason="low_confidence"`,
and everything in `dropped`. When nothing survives, `primary = UNKNOWN_INTENT`,
`confidence = 0.0`, `multi_intent = False`, `reason = "ok"`.

Build each `IntentMatch` from the surviving `IntentScore`, carrying `min_score`
and `rule_hits` built from its `RawHit`s.

- [ ] **Step 4: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: all passed

- [ ] **Step 5: Commit** (conditional)

```bash
git add src/intent/matching.py tests/test_pipeline.py
git commit -m "feat: deterministic ranking, multi-intent cap, confidence gate"
```

---

## Task 8: The rule content, and the fixtures

The JSON now exists and the spec fixtures (§13) pin it. This is the task where
"does it actually work" gets answered.

**Files:**
- Create: `src/intent/rules/pt.json`, `src/intent/rules/en.json`,
  `tests/fixtures/pt_cases.json`, `tests/test_rules_content.py`
- Modify: `Makefile` (add the `demo` target)

**Interfaces:**
- Consumes: `load_ruleset` (Task 4), `Matcher` and `rank` (Tasks 5–7).
- Produces: two rule documents, each with the same eleven rule-driven ids from
  spec §7, plus `negators` and `negation_boundaries`.
- Constraint carried from spec §4.1: **`all` values are single tokens**, and
  `pattern` values are written accent-free.

- [ ] **Step 1: Write the failing fixture-driven test**

`tests/fixtures/pt_cases.json`:

```json
{
  "cases": [
    { "text": "quero cancelar meu plano", "expected": "cancelar" },
    { "text": "como faço para cancelar a assinatura?", "expected": "cancelar" },
    { "text": "quero assinar o plano pro", "expected": "comprar" },
    { "text": "a página não carrega e dá erro", "expected": "suporte" },
    { "text": "oi, tudo bem?", "expected": "saudacao" },
    { "text": "quero sair!!!", "expected": "cancelar" },
    { "text": "não quero cancelar, quero saber do reembolso", "expected": "pagamento" },
    { "text": "o problema é que não consigo acessr", "expected": "desconhecido" },
    { "text": "quero cancelar mas mudei de ideia, na verdade quero um plano maior",
      "expected": "upgrade" },
    { "text": "quero cancelar e depois assinar o plano Pro",
      "expected": "cancelar", "also_expected": ["comprar"] }
  ]
}
```

```python
import json
import pathlib

import pytest
import spacy

from intent.config import Settings
from intent.matching import UNKNOWN_INTENT, Matcher, rank
from intent.normalize import normalise
from intent.rules import load_ruleset

CASES = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "pt_cases.json")
    .read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        return spacy.blank("pt")


@pytest.fixture(scope="module")
def engine(nlp):
    ruleset = load_ruleset("pt")
    settings = Settings.from_env(env={})
    matcher = Matcher(ruleset, settings)
    return lambda text: rank(matcher.evaluate(nlp(text), normalise(text)), settings)


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:34] for c in CASES])
def test_pt_fixture(engine, case):
    result = engine(case["text"])
    assert result.primary == case["expected"], (
        f"{case['text']!r} -> got {result.primary}; "
        f"scores: {[(s.intent_id, s.score) for s in result.dropped]}")
    for other in case.get("also_expected", []):
        assert other in [m.intent for m in result.kept]
        assert result.multi_intent is True


def test_every_catalogued_intent_has_rules_and_examples():
    ruleset = load_ruleset("pt")
    assert {i.id for i in ruleset.intents} == {
        "cancelar", "reclamacao", "pagamento", "acesso", "upgrade", "comprar",
        "downgrade", "suporte", "alterar_dados", "duvida", "saudacao"}
    for intent in ruleset.intents:
        assert intent.rules, intent.id
        assert intent.examples, intent.id


def test_priorities_match_the_spec_catalogue():
    expected = {"cancelar": 90, "reclamacao": 80, "pagamento": 75, "acesso": 70,
                "upgrade": 65, "comprar": 60, "downgrade": 55, "suporte": 50,
                "alterar_dados": 45, "duvida": 40, "saudacao": 20}
    for intent in load_ruleset("pt").intents:
        assert intent.priority == expected[intent.id], intent.id


def test_pt_and_en_expose_the_same_intent_ids():
    assert ({i.id for i in load_ruleset("pt").intents}
            == {i.id for i in load_ruleset("en").intents})


def test_comprar_does_not_fire_on_desassinar(nlp):
    # Review Focus #2, against the real rules rather than a fixture ruleset
    settings = Settings.from_env(env={})
    matcher = Matcher(load_ruleset("pt"), settings)
    result = rank(matcher.evaluate(nlp("quero desassinar o contrato"),
                                   normalise("quero desassinar o contrato")),
                  settings)
    assert "comprar" not in [m.intent for m in result.kept]
```

The fixture test asserts the **primary** only. Asserting that no other intent
survives would over-specify: `"como faço para cancelar a assinatura?"` is
legitimately both a `duvida` and a `cancelar`, and the engine is right to say so.

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rules_content.py -v`
Expected: `UnknownLanguageError` — `rules/pt.json` does not exist yet

- [ ] **Step 3: Write `src/intent/rules/pt.json``

Priorities and `min_score` exactly as pinned by `test_priorities_match_the_spec_catalogue`
and spec §7. The `min_score` values are the ones the spec fixes: `cancelar` 3.0,
`reclamacao` 3.0, `pagamento` 2.5, `acesso` 2.5, `upgrade` 3.0, `comprar` 2.0,
`downgrade` 3.0, `suporte` 2.5, `alterar_dados` 3.0, `duvida` 2.5, `saudacao` 1.0.

The rules below are the minimum set the fixtures require. Weights are not
arbitrary; each line says why it has the value it has.

**`cancelar`** (min 3.0) — a bare `keyword cancelar` scores 2.0 and is therefore
*not* enough on its own. That is deliberate: `min_score` 3.0 is the spec's
false-positive guard, so a cancellation needs a phrase or a second signal.

| type | value | weight | why |
|---|---|---|---|
| keyword | `cancelar` | 2.0 | the base word |
| pattern | `(quero\|queria\|preciso\|gostaria)\s+(cancelar\|encerrar\|dispensar)` | 3.0 | so "quero cancelar" alone clears 3.0 — a user typing only that must not fall through to a human |
| pattern | `cancelar\s+(meu\s+)?(plano\|assinatura\|conta\|servico\|contrato)` | 3.0 | the object makes it unambiguous |
| pattern | `(sair\|deixar o servico\|encerrar a assinatura\|dar baixa)` | 3.0 | fixture 6; also proves punctuation does not block matching |
| all | `["cancelar", "assinatura"]` | 3.5 | two independent signals |

**`upgrade`** (min 3.0) — the reversal phrase is the one rule in the file with a
weight no ordinary phrase would need, and spec fixture 4 is the reason.

| type | value | weight | why |
|---|---|---|---|
| pattern | `(mudei de ideia\|na verdade\|deixa pra tras\|desisti)` | **6.0** | a reversal phrase overrides whatever came before; 6.0 carries `upgrade` past `cancelar` on its own |
| pattern | `(mudar para o plano\|upgrade\|plano premium)` | 3.0 | the request itself. Do **not** re-add `plano maior` here: the `all` below consumes both tokens, so that alternative could never score |
| all | `["plano", "maior"]` | 3.5 | two independent tokens; this is what matches "plano maior" |

**`comprar`** (min 2.0) — `keyword assinar` 2.0 and
`pattern (assinar\|contratar\|adquirir\|quero um plano\|novo plano)` 3.0. Both
fire on "assinar", which is correct: a phrase is stronger evidence than a word.
Neither fires on "desassinar", because the keyword compares whole tokens and the
pattern has implicit word boundaries.

**`suporte`** (min 2.5) — `pattern nao\s+(funciona|abre|carrega|responde)` 2.5,
`keyword suporte` 2.0, `lemma erro` 1.0.

**`pagamento`** (min 2.5) —
`pattern (reembolso\|estorno\|cobranca\|boleto\|pix\|cartao\|fatura\|cobrou)` 2.5
and `all ["duas", "vezes"]` 3.5. This is what fixture 7 needs: `reembolso`
matches un-negated for 2.5, which clears `min_score`, while `cancelar` is
negated to 0.0 and disappears entirely.

**`acesso`** (min 2.5) — `pattern (nao consigo (entrar|acessar)|login|senha|2fa|conta bloqueada)` 2.5.
No fuzzy matching, so fixture 8's `acessr` matches nothing — that is the
assertion, not an oversight.

**`reclamacao`** (min 3.0) — `pattern (reclamacao\|reclamar\|pior atendimento\|nao aceito\|exijo)` 3.0,
`keyword reclamacao` 2.0.

**`downgrade`** (min 3.0) — `pattern (plano menor\|downgrade\|reduzir o plano\|economizar)` 3.0.

**`alterar_dados`** (min 3.0) —
`pattern (alterar|atualizar|trocar)\s+(meu\s+)?(nome\|email\|e-mail\|endereco\|documento)` 3.0,
`all ["meu", "email"]` 3.5.

**`duvida`** (min 2.5) — `pattern (como funciona\|duvida\|qual a diferenca\|como faco para)` 2.5.

**`saudacao`** (min 1.0) — `pattern ^(ola\|oi\|bom dia\|boa tarde\|boa noite\|e ai)` 1.5,
`keyword ola` 1.0. Anchored so a greeting cannot fire mid-sentence. Every rule
**value** in both catalogues is accent-free: the engine strips accents before
matching, and a hand-edited file with one accented value in it is a trap for the
next author. `negators` and `negation_boundaries` are the exception — they keep
correct Portuguese spelling, because `matching.py` folds accents when it compares
them.

`negators`: `não`, `nunca`, `jamais`, `sem`, `deixar de`, `sem querer`,
`prefiro não`, `não pretendo`.
`negation_boundaries`: `mas`, `porém`, `porque`, `pois`, `então`, `e`.

Work fixture 4 by hand before running the tests. "quero cancelar mas mudei de
ideia, na verdade quero um plano maior": `cancelar` collects 2.0 (keyword) + 3.0
(quero…cancelar) + 0 (`all ["cancelar","assinatura"]` — no "assinatura") = 5.0.
`upgrade` collects 6.0 (reversal) + 0 (the request pattern: its tokens are the
same two the `all` already consumed) + 3.5 (`all`) = 9.5. `upgrade` wins. Note
`cancelar` is *not* negated: the reversal words come after it and negation only
ever looks left. This is also the worked example for why an `all` rule shadows an
overlapping pattern — check the token indices, not just the weights, when
hand-calculating a score.

- [ ] **Step 4: Write `src/intent/rules/en.json`**

The same eleven ids with English values and labels, `en` negators
(`not`, `never`, `no longer`, `stop`, `without`) and boundaries
(`but`, `because`, `and`, `so`). Identical priorities and `min_score`, so the
two catalogues stay comparable in the UI's expected-vs-detected table.

Add this to `tests/test_rules_content.py` while you are in the file. The validator
accepts `all ["cancelar", "Cancelar"]`, which scores full weight for one mention
of one word — a silent bypass of §4.3's "two independent signals". Catch it in the
content, where the mistake is actually made, rather than reopening Task 4's
validator for a rule the shipped catalogues do not violate:

```python
@pytest.mark.parametrize("language", ["pt", "en"])
def test_all_rules_name_two_distinct_signals(language):
    ruleset = load_ruleset(language)
    for intent in ruleset.intents:
        for rule in intent.rules:
            if rule.type is not RuleType.ALL:
                continue
            folded = [strip_accents(v.lower()) for v in rule.value]
            assert len(set(folded)) == len(folded), f"{rule.rule_id}: {rule.value}"
```

The same reasoning applies to the `upgrade` shadowing fixed above: when a pattern
and an `all` cover the same tokens, the `all` wins by construction, so check
token overlap before adding a rule whose value looks like another rule's.

- [ ] **Step 5: Run the fixtures and read the failure messages**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rules_content.py -v`
Expected: some cases fail. The assertion message prints the text, the value
obtained, and every dropped intent with its score. That is enough to tell
whether the fix is a missing rule, a `min_score` that is too high, or a negation
that reached too far.

**Do not lower a `min_score` to make a case pass.** That erases the
false-positive guard the whole design rests on. Add a rule, or fix a weight.

- [ ] **Step 6: Add the `demo` target to the Makefile**

```makefile
demo:
	$(PY) -m intent.cli demo
```

`intent.cli` arrives in Task 13, so this target fails until then. Do not create
a stub CLI to satisfy it.

- [ ] **Step 7: Invert Task 4's two default-path tests**

Task 4 could not load a real rules file, so it pinned the default path by
asserting the file was *absent*. Now that `pt.json` and `en.json` exist, both of
those tests are guaranteed-red. Rewrite them as positive load tests, keeping the
same intent — the default path resolves in-package, not under `resources/`:

```python
def test_default_path_loads_the_shipped_ruleset():
    ruleset = load_ruleset("pt")
    assert ruleset.language == "pt"
    assert ruleset.intents


def test_load_all_rulesets_loads_every_configured_language():
    rulesets = load_all_rulesets(("pt", "en"))
    assert set(rulesets) == {"pt", "en"}
    assert all(r.intents for r in rulesets.values())
```

The `assert "resources" not in named` line from Task 4 is dropped along with the
absent-file assertion it guarded; in its place, assert the resolved path sits
beside `rules.py` rather than under the repo's `resources/` directory.

- [ ] **Step 8: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: everything written so far passes, including the two rewritten tests
from Step 7.

- [ ] **Step 9: Commit** (conditional)

```bash
git add src/intent/rules tests/fixtures tests/test_rules_content.py Makefile
git commit -m "feat: pt and en rule content, fixtures pin the behaviour"
```

---

## Task 9: spaCy loading with fallback, and result assembly

**Files:**
- Create: `src/intent/pipeline.py`
- Modify: `tests/test_pipeline.py` (extend with the tests below)
- Modify: `tests/conftest.py` (add the `engine_with_blank` fixture)

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces:
  - `Pipeline(rulesets: dict[str, RuleSet], profiles: dict, settings: Settings,
    nlp_by_language: dict[str, Any] | None = None)`. `nlp_by_language` is the
    injection seam that lets tests pin a blank pipe and never depend on whether
    a model is installed.
  - `Pipeline.detect(text: str) -> IntentResult`
  - `Pipeline.detect_batch(texts: list[str]) -> list[IntentResult]` — identifies
    first, then one `nlp.pipe` call per language group, reassembled in input
    order.
  - `intent.pipeline.explain(result: IntentResult) -> str` — the PT-BR sentence,
    a module-level function so it is testable without a `Pipeline`.

- [ ] **Step 1: Add the fixture to `tests/conftest.py`**

```python
from intent.language import load_profiles
from intent.pipeline import Pipeline
from intent.rules import load_all_rulesets


@pytest.fixture
def blank_pipes():
    return {"pt": spacy.blank("pt"), "en": spacy.blank("en")}


@pytest.fixture
def engine_with_blank(settings, blank_pipes):
    return Pipeline(rulesets=load_all_rulesets(settings.languages),
                    profiles=load_profiles(),
                    settings=settings,
                    nlp_by_language=blank_pipes)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
from intent.errors import InputError
from intent.pipeline import Pipeline, explain


def test_empty_and_short_text_are_client_errors(engine_with_blank):
    for text in ("", "   ", "a"):
        with pytest.raises(InputError):
            engine_with_blank.detect(text)


def test_punctuation_only_is_empty_text_not_an_error(engine_with_blank):
    # long enough to pass the length gate, with nothing to reason about
    r = engine_with_blank.detect("!!!")
    assert r.language.reason == "empty_text"
    assert r.primary == "desconhecido"
    assert r.intents == ()


def test_too_long_is_a_client_error(engine_with_blank):
    with pytest.raises(InputError):
        engine_with_blank.detect("a" * 5001)


def test_detect_returns_the_expected_intent(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar meu plano")
    assert r.primary == "cancelar"
    assert r.rules_version == "1.0.0"


def test_the_explanation_is_portuguese_and_names_the_rules(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar meu plano")
    assert r.explanation == explain(r)
    assert "cancelar" in r.explanation.lower()


def test_tokens_expose_clean_names(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar meu plano")
    assert r.tokens
    assert any(t.matched for t in r.tokens)
    assert not any(hasattr(t, "lemma_") for t in r.tokens)


def test_negated_tokens_are_marked(engine_with_blank):
    r = engine_with_blank.detect("não quero cancelar")
    assert any(t.negated for t in r.tokens)
    assert r.primary == "desconhecido"


def test_the_resource_report_names_the_model_in_use(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar")
    assert r.resources.models["pt"]["fallback"] is True
    assert r.resources.models["pt"]["model"] == "blank(pt)"


def test_the_resource_report_carries_the_rules_digest(engine_with_blank):
    r = engine_with_blank.detect("quero cancelar")
    assert r.resources.rules["pt"]["version"] == "1.0.0"
    assert len(r.resources.rules["pt"]["sha256"]) == 64


def test_an_unsupported_language_is_reported_not_guessed(engine_with_blank):
    r = engine_with_blank.detect("취소해줘")
    assert r.language.supported is False
    assert r.language.reason == "unsupported_script"
    assert r.primary == "desconhecido"
    assert r.intents == ()


def test_batch_preserves_input_order(engine_with_blank):
    texts = ["quero cancelar meu plano", "취소해줘", "oi, tudo bem?",
             "quero assinar o plano pro"]
    assert [r.text for r in engine_with_blank.detect_batch(texts)] == texts


def test_batch_agrees_with_single_detection(engine_with_blank):
    texts = ["quero cancelar meu plano", "quero assinar o plano pro",
             "i want to cancel my subscription"]
    batch = engine_with_blank.detect_batch(texts)
    singles = [engine_with_blank.detect(t) for t in texts]
    assert [r.primary for r in batch] == [r.primary for r in singles]


def test_batch_over_the_limit_is_rejected(engine_with_blank):
    with pytest.raises(InputError):
        engine_with_blank.detect_batch(["oi"] * 501)


def test_batch_of_zero_is_fine(engine_with_blank):
    assert engine_with_blank.detect_batch([]) == []


def test_duration_is_measured(engine_with_blank):
    assert engine_with_blank.detect("quero cancelar").duration_ms >= 0.0
```

- [ ] **Step 3: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: `ImportError: cannot import name 'Pipeline'`

- [ ] **Step 4: Implement `pipeline.py`**

`detect`:

1. `text.strip()`; raise `InputError` when it is empty, shorter than
   `settings.min_text_chars`, or longer than `settings.max_text_chars`.
2. `n = normalise(text)`; `info = identifier.identify(text)`.
3. If `not info.supported`, return an `IntentResult` with `intents=()`,
   `primary="desconhecido"`, and that `info` — **no spaCy call at all**, since
   there is no pipeline to call.
4. `doc = self._nlp(info.language)(text)`;
   `result = rank(matcher.evaluate(doc, n), settings)`.
5. Build `TokenView`s from `doc`, setting `matched`/`negated` from the union of
   the trace's `token_indices`. `lemma` and `pos` come from `token.lemma_` and
   `token.pos_`; keep those accesses inside this function.
6. `ResourceReport` records, per language, the model name or
   `f"blank({language})"` with `fallback`, and `version`/`entries`/`sha256` from
   the `RuleSet`. Add a `limitations` entry for each language whose active
   pipeline cannot lemmatise while its ruleset contains `lemma` rules.
7. `duration_ms` from `time.perf_counter`.

`_nlp(language)`: use the injected map when given, else `spacy.load` the
corresponding model, else `spacy.blank(language)` plus `add_pipe("sentencizer")`.
A missing model is a `limitations` entry, never a raised error (§6.3).

`detect_batch`: identify every text, group indices by language, run one
`nlp.pipe` per group, evaluate, reassemble in the original order. Raise
`InputError` above `settings.batch_max_items`.

`explain`: PT-BR, naming the intent label and the rules that fired — for example
`"Mensagem classificada como Cancelar assinatura (regras: keyword 'cancelar',
pattern 'quero … cancelar')."`. For `desconhecido` caused by an unsupported
language, name the reason instead of a rule.

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: all passed

- [ ] **Step 6: Commit** (conditional)

```bash
git add src/intent/pipeline.py tests/conftest.py tests/test_pipeline.py
git commit -m "feat: spaCy fallback loading and IntentResult assembly"
```

---

## Task 10: Service layer and the conversation store

**Files:**
- Create: `src/intent/service.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `Pipeline` (Task 9), `Settings`, `schemas`.
- Produces:
  - `IntentService(pipeline, settings, store=None)` with
    `detect(text, require_supported=False) -> IntentResult`,
    `detect_batch(texts, require_supported=False) -> list[IntentResult]`,
    `catalogue(language=None) -> list[dict]`,
    `rules_document(language) -> dict`,
    `validate_rules_document(document) -> dict`,
    `health() -> dict`, `conversations(limit=20) -> list[dict]`.
  - `ConversationStore(db_path: pathlib.Path)` with
    `save_conversation(turn: dict) -> None` and `recent(limit: int) -> list[dict]`.
  - `require_supported` raises `UnsupportedLanguageError` **only** when
    `info.supported is False`, never on low intent confidence (Review Focus #4).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from intent.errors import InputError, RulesError, UnsupportedLanguageError
from intent.service import ConversationStore, IntentService


def test_require_supported_raises_only_for_an_unsupported_language(service):
    with pytest.raises(UnsupportedLanguageError) as exc:
        service.detect("취소해줘", require_supported=True)
    assert exc.value.details["reason"] == "unsupported_script"
    assert exc.value.details["detected_language"] is None


def test_require_supported_does_not_raise_on_low_confidence(service):
    # Review Focus #4: the language was identified, the request is valid
    result = service.detect("o problema é que não consigo acessr",
                            require_supported=True)
    assert result.primary == "desconhecido"
    assert result.language.supported is True


def test_detect_without_require_supported_returns_the_result(service):
    result = service.detect("취소해줘")
    assert result.language.supported is False
    assert result.primary == "desconhecido"


def test_catalogue_lists_every_intent_with_a_rule_count(service):
    catalogue = service.catalogue("pt")
    assert len(catalogue) == 11
    cancelar = next(c for c in catalogue if c["id"] == "cancelar")
    assert cancelar["priority"] == 90
    assert cancelar["min_score"] == 3.0
    assert cancelar["rule_count"] >= 3
    assert cancelar["examples"]


def test_catalogue_without_a_language_uses_the_first_configured(service):
    assert service.catalogue() == service.catalogue("pt")


def test_rules_document_round_trips(service):
    document = service.rules_document("pt")
    assert document["language"] == "pt"
    assert service.validate_rules_document(document) == {"valid": True}


def test_validate_rejects_a_broken_document(service):
    with pytest.raises(RulesError):
        service.validate_rules_document(
            {"language": "pt", "version": "x", "intents": []})


def test_health_reports_models_versions_and_limitations(service):
    health = service.health()
    assert health["status"] == "ok"
    assert set(health["rules_versions"]) == {"pt", "en"}
    assert set(health["models"]) == {"pt", "en"}
    assert isinstance(health["limitations"], list)


def test_no_store_means_no_conversations(service):
    assert service.store is None
    assert service.conversations() == []


def test_the_store_round_trips_a_turn(service_with_store):
    service_with_store.store.save_conversation({
        "session_id": "s1", "turn_index": 1, "user_message": "oi",
        "bot_reply": "olá!", "primary": "saudacao"})
    rows = service_with_store.conversations(limit=5)
    assert rows[0]["primary"] == "saudacao"
    assert rows[0]["user_message"] == "oi"


def test_the_store_returns_the_newest_first(service_with_store):
    store = service_with_store.store
    for index in range(3):
        store.save_conversation({"session_id": "s1", "turn_index": index,
                                 "user_message": f"m{index}", "bot_reply": "r",
                                 "primary": "saudacao"})
    assert [r["user_message"] for r in store.recent(limit=3)] == ["m2", "m1", "m0"]


def test_batch_over_the_limit_is_rejected_at_the_service_boundary(service):
    with pytest.raises(InputError):
        service.detect_batch(["oi"] * 501)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_service.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.service'`

- [ ] **Step 3: Implement `service.py`**

`IntentService` takes its collaborators as constructor arguments.

`ConversationStore` creates its parent directory on construction, uses
`sqlite3`, and inserts rows newest-first on read. It stores the message already
redacted: redaction belongs to `Chatbot` (Task 11), and a store that redacted
would be a second place to forget. Say so in a comment where the insert happens,
because it is the kind of note a future editor deletes as obvious.

- [ ] **Step 4: Add the service fixtures to `tests/conftest.py`**

```python
@pytest.fixture
def service(engine_with_blank, settings):
    return IntentService(pipeline=engine_with_blank, settings=settings)


@pytest.fixture
def service_with_store(engine_with_blank, settings, tmp_path):
    return IntentService(pipeline=engine_with_blank, settings=settings,
                         store=ConversationStore(tmp_path / "t.db"))
```

`settings` and `engine_with_blank` are already there from Tasks 5 and 9; do not
redefine them.

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_service.py -v`
Expected: all passed

- [ ] **Step 6: Commit** (conditional)

```bash
git add src/intent/service.py tests/conftest.py tests/test_service.py
git commit -m "feat: service use cases and optional conversation store"
```

---

## Task 11: The chatbot

**Files:**
- Create: `src/intent/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `IntentService`, `Settings`.
- Produces:
  - `ChatResponse(reply, reply_format, session_id, intents, primary,
    multi_intent, strategy, confidence, turn_index, pending_step)` — frozen.
  - `Chatbot(service, settings, plans=None)` with
    `responder(message, session_id=None, require_supported=False) -> ChatResponse`.
  - `redact(text: str) -> tuple[str, list[str]]` — the redacted text plus the
    names of the patterns that fired.
  - `SENSITIVE_PATTERNS` — three named compiled patterns: card-like digit runs
    (13–19 digits, optionally separated by spaces or hyphens), a
    password-assignment phrase, and a bearer-token shape.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from intent.chat import SENSITIVE_PATTERNS, redact


def test_greeting_gets_a_canned_reply(chatbot):
    r = chatbot.responder("oi, tudo bem?")
    assert r.primary == "saudacao"
    assert r.reply
    assert r.reply_format == "text"
    assert r.session_id
    assert r.turn_index == 1


def test_unknown_routes_to_a_human(chatbot):
    r = chatbot.responder("o problema é que não consigo acessr")
    assert r.primary == "desconhecido"
    assert "atendente" in r.reply.lower()


def test_an_unsupported_language_still_gets_a_reply(chatbot):
    r = chatbot.responder("취소해줘")
    assert r.reply
    assert r.primary == "desconhecido"


def test_multi_intent_produces_one_coherent_reply(chatbot):
    r = chatbot.responder("quero cancelar e depois assinar o plano Pro")
    assert r.multi_intent is True
    assert r.primary == "cancelar"
    assert "comprar" in r.intents


def test_the_collector_advances_one_step_per_turn(chatbot):
    first = chatbot.responder("quero cancelar meu plano")
    assert first.pending_step == 1
    second = chatbot.responder("sim", first.session_id)
    assert second.pending_step == 2
    third = chatbot.responder("está caro", first.session_id)
    assert third.pending_step == 3
    assert "protocolo" in third.reply.lower()


def test_the_collector_is_abandoned_when_the_user_changes_mind(chatbot):
    # spec §10 routing step 3: a collector that swallows every later turn
    # silently discards what the user actually asked
    first = chatbot.responder("quero cancelar meu plano")
    second = chatbot.responder("na verdade quero um plano maior", first.session_id)
    assert second.primary == "upgrade"
    assert second.pending_step is None


def test_the_turn_index_increments_within_a_session(chatbot):
    a = chatbot.responder("oi")
    b = chatbot.responder("tudo bem?", a.session_id)
    assert b.turn_index == a.turn_index + 1


def test_a_card_number_is_redacted_and_never_echoed(chatbot):
    r = chatbot.responder("meu cartão é 4111 1111 1111 1111 e quero cancelar")
    assert "4111" not in r.reply
    assert "[REDACTED]" in r.reply or "atendente" in r.reply.lower()


def test_the_bot_never_asks_for_a_password(chatbot):
    r = chatbot.responder("não consigo entrar na minha conta")
    low = r.reply.lower()
    assert "digite sua senha" not in low
    assert "informe sua senha" not in low
    assert "envie sua senha" not in low


def test_redact_reports_what_it_caught():
    text, caught = redact("minha senha é hunter2 e o cartão 4111111111111111")
    assert "hunter2" not in text
    assert "4111111111111111" not in text
    assert caught


def test_redact_leaves_ordinary_text_alone():
    text, caught = redact("quero cancelar meu plano")
    assert text == "quero cancelar meu plano"
    assert caught == []


def test_every_sensitive_pattern_is_named():
    # a pattern nobody tested is a pattern that does not work
    assert len(SENSITIVE_PATTERNS) == 3
    assert {name for name, _ in SENSITIVE_PATTERNS} == {"cartao", "senha", "bearer"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_chat.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.chat'`

- [ ] **Step 3: Implement `chat.py`**

`redact` runs each named pattern over the text and replaces the span with
`[REDACTED]`, collecting the names. Apply it to the user's message **before**
storing it or including it in a reply, and leave a comment saying that detection
still runs on the original text — redacting first would blind the rules to
`senha` and `cartão`, which are exactly the words `acesso` and `pagamento` match
on.

Sessions live in an in-memory dict keyed by `session_id`, a `uuid4().hex` when
none is given. Routing, per spec §10:

1. `result = service.detect(redacted_message, require_supported=...)`
2. If `session.pending_intent` is set and `result.primary == "desconhecido"`,
   advance the collector and return.
3. Else if `result.primary` differs from `session.pending_intent` and is a real
   intent, clear `pending_intent`/`pending_step` and answer the new intent.
4. Else answer `result.primary`.

`strategy` is the intent id for a plain answer, or
`f"collector:{pending_intent}"` for a collector turn. One strategy function per
intent, from the table in spec §10, all returning PT-BR. The `acesso` strategy
must contain no imperative asking the user to type a password.

- [ ] **Step 4: Add the `chatbot` fixture to `tests/conftest.py`**

```python
from intent.chat import Chatbot


@pytest.fixture
def chatbot(service, settings):
    return Chatbot(service=service, settings=settings)
```

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_chat.py -v`
Expected: all passed

- [ ] **Step 6: Commit** (conditional)

```bash
git add src/intent/chat.py tests/conftest.py tests/test_chat.py
git commit -m "feat: rules-based chatbot with collectors, routing and PII redaction"
```

---

## Task 12: HTTP API

**Files:**
- Create: `src/intent/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `IntentService` (Task 10), `Chatbot` (Task 11), `Errors`.
- Produces:
  - `create_app(service=None, settings=None, chatbot=None) -> FastAPI` — the
    injection seam. When `service` or `settings` is omitted, build from
    `Settings.from_env()` and the real pipeline. When `chatbot` is omitted,
    construct `Chatbot(service, settings)`, so the chat route works without the
    caller doing anything.
  - `get_app() -> FastAPI` — the ASGI entrypoint `app.py` imports.
  - Routes: `POST /api/v1/detect`, `POST /api/v1/detect/batch`,
    `GET /api/v1/intents`, `GET /api/v1/rules/{lang}`,
    `POST /api/v1/rules/validate`, `GET /api/v1/health`,
    `GET /api/v1/conversations`, `POST /api/v1/chat`, plus `/`, `/index.html`,
    `/styles.css` and `/js/{path}` from `web/`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(app_with_service):
    return TestClient(app_with_service)


def test_detect_returns_200(client):
    r = client.post("/api/v1/detect", json={"text": "quero cancelar meu plano"})
    assert r.status_code == 200
    assert r.json()["primary"] == "cancelar"


def test_detect_missing_body_is_422_with_the_envelope(client):
    r = client.post("/api/v1/detect", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "input_invalid"
    assert r.json()["error"]["retryable"] is False


def test_detect_blank_text_is_422(client):
    r = client.post("/api/v1/detect", json={"text": "   "})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "input_invalid"


def test_detect_over_max_length_is_422(client):
    r = client.post("/api/v1/detect", json={"text": "a" * 5001})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "input_invalid"


def test_an_unsupported_language_is_200_by_default(client):
    r = client.post("/api/v1/detect", json={"text": "취소해줘"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"]["supported"] is False
    assert body["language"]["reason"] == "unsupported_script"


def test_require_supported_makes_an_unsupported_language_a_422(client):
    r = client.post("/api/v1/detect",
                    json={"text": "취소해줘", "require_supported": True})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_language"
    assert r.json()["error"]["details"]["reason"] == "unsupported_script"


def test_require_supported_is_200_for_a_supported_language_with_no_intent(client):
    # Review Focus #4
    r = client.post("/api/v1/detect",
                    json={"text": "o problema é que não consigo acessr",
                          "require_supported": True})
    assert r.status_code == 200
    assert r.json()["primary"] == "desconhecido"


def test_batch_preserves_order(client):
    texts = ["quero cancelar meu plano", "취소해줘", "oi, tudo bem?"]
    r = client.post("/api/v1/detect/batch", json={"texts": texts})
    assert r.status_code == 200
    assert [x["text"] for x in r.json()["results"]] == texts


def test_batch_over_the_limit_is_422(client):
    r = client.post("/api/v1/detect/batch", json={"texts": ["oi"] * 501})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "input_invalid"


def test_the_intents_catalogue(client):
    r = client.get("/api/v1/intents", params={"language": "pt"})
    assert r.status_code == 200
    assert len(r.json()["intents"]) == 11


def test_rules_document_and_unknown_language(client):
    assert client.get("/api/v1/rules/pt").status_code == 200
    r = client.get("/api/v1/rules/xx")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_language"


def test_rules_validate_accepts_good_and_rejects_bad(client):
    good = client.get("/api/v1/rules/pt").json()
    assert client.post("/api/v1/rules/validate", json=good).status_code == 200
    bad = client.post("/api/v1/rules/validate",
                      json={"language": "pt", "version": "x"})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "invalid_rules"


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_the_chat_endpoint(client):
    r = client.post("/api/v1/chat", json={"message": "quero cancelar meu plano"})
    assert r.status_code == 200
    body = r.json()
    assert body["primary"] == "cancelar"
    assert body["session_id"]
    assert body["reply"]


def test_the_chat_endpoint_keeps_the_session(client):
    first = client.post("/api/v1/chat", json={"message": "quero cancelar meu plano"}).json()
    second = client.post("/api/v1/chat", json={
        "message": "sim", "session_id": first["session_id"]}).json()
    assert second["turn_index"] == 2


def test_conversations_is_empty_without_a_store(client):
    r = client.get("/api/v1/conversations")
    assert r.status_code == 200
    assert r.json()["conversations"] == []


def test_the_error_envelope_has_the_same_keys_for_every_code(client):
    r = client.get("/api/v1/rules/xx")
    assert set(r.json()["error"]) == {"code", "message", "details", "retryable"}


def test_the_static_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.api'`

- [ ] **Step 3: Implement `api.py`**

One exception handler per `IntentError` subclass, registered on the app,
emitting `{"error": {"code", "message", "details", "retryable"}}`. Status
mapping: `InputError` and `RulesError` → 422; `UnknownLanguageError` and
`UnknownIntentError` → 404; `UnsupportedLanguageError` → 422;
`ModelUnavailableError` and `StorageError` → 503; bare `IntentError` → 500.

Request models, Pydantic v2: `DetectRequest(text: str, require_supported: bool =
False)`, `BatchRequest(texts: list[str], require_supported: bool = False)`,
`ChatRequest(message: str, session_id: str | None = None, require_supported:
bool = False)`.

Serialise the frozen dataclasses through one `asdict`-based helper rather than a
`from_attributes` model per view, so a field added to `schemas.py` reaches the
API without a second edit.

Static files: resolve the web root as `Path(__file__).resolve().parents[2] /
"web"`, mount `/js`, and serve `index.html` at `/` and `/index.html` and
`styles.css` at `/styles.css`.

- [ ] **Step 4: Add the `app_with_service` fixture to `tests/conftest.py`**

```python
from intent.api import create_app


@pytest.fixture
def app_with_service(service, settings):
    return create_app(service=service, settings=settings)
```

`create_app` builds the `Chatbot` itself, so there is nothing else to wire.

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v`
Expected: all passed

- [ ] **Step 6: Verify `app.py` now imports**

Run: `.\.venv\Scripts\python.exe -c "import app; print(app.app.title)"`
Expected: prints the FastAPI title. A failure here means static-file mounting or
handler registration is broken in a way the route tests did not catch.

- [ ] **Step 7: Commit** (conditional)

```bash
git add src/intent/api.py app.py tests/conftest.py tests/test_api.py
git commit -m "feat: REST API with a uniform error envelope"
```

---

## Task 13: CLI

**Files:**
- Create: `src/intent/cli.py`, `tests/test_cli.py`
- Modify: `Makefile` (add `rules-validate`)

**Interfaces:**
- Consumes: `create_app`, `IntentService`, `Pipeline`, `Chatbot`, `Settings`.
- Produces: `intent.cli.app` — a Typer app named `intent` with the commands
  `detect`, `batch`, `list`, `rules show`, `rules validate`, `demo`, `chat`,
  `serve`, and `intent.cli.main()` as the console-script entrypoint.

- [ ] **Step 1: Write the failing tests**

```python
import json

from typer.testing import CliRunner

from intent.cli import app

runner = CliRunner()


def test_detect_identified_exits_0():
    r = runner.invoke(app, ["detect", "quero cancelar meu plano"])
    assert r.exit_code == 0
    assert "cancelar" in r.stdout


def test_detect_unknown_exits_1():
    r = runner.invoke(app, ["detect", "o problema é que não consigo acessr"])
    assert r.exit_code == 1
    assert "desconhecido" in r.stdout


def test_detect_json_output_is_parseable():
    r = runner.invoke(app, ["detect", "--json", "quero cancelar meu plano"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["primary"] == "cancelar"


def test_detect_blank_exits_2():
    assert runner.invoke(app, ["detect", "   "]).exit_code == 2


def test_detect_over_max_length_exits_2():
    assert runner.invoke(app, ["detect", "a" * 5001]).exit_code == 2


def test_detect_reads_stdin():
    r = runner.invoke(app, ["detect", "--stdin"], input="quero cancelar meu plano\n")
    assert r.exit_code == 0
    assert "cancelar" in r.stdout


def test_list_shows_the_catalogue():
    r = runner.invoke(app, ["list"])
    assert r.exit_code == 0
    assert "cancelar" in r.stdout
    assert "suporte" in r.stdout


def test_rules_show_and_validate():
    assert runner.invoke(app, ["rules", "show", "pt"]).exit_code == 0
    r = runner.invoke(app, ["rules", "validate", "src/intent/rules/pt.json"])
    assert r.exit_code == 0
    assert "válido" in r.stdout


def test_rules_validate_rejects_a_broken_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"language": "pt", "version": "x", "intents": []}',
                   encoding="utf-8")
    assert runner.invoke(app, ["rules", "validate", str(bad)]).exit_code == 2


def test_rules_show_unknown_language_exits_2():
    assert runner.invoke(app, ["rules", "show", "xx"]).exit_code == 2


def test_batch_reads_a_file(tmp_path):
    path = tmp_path / "in.json"
    path.write_text(json.dumps(["quero cancelar meu plano", "oi, tudo bem?"]),
                    encoding="utf-8")
    r = runner.invoke(app, ["batch", str(path)])
    assert r.exit_code == 0
    assert "cancelar" in r.stdout


def test_demo_reports_expected_against_detected():
    r = runner.invoke(app, ["demo"])
    assert "esperado" in r.stdout.lower()
    assert "detectado" in r.stdout.lower()


def test_demo_fails_when_a_fixture_disagrees():
    # if demo cannot fail, it is decoration rather than a check
    r = runner.invoke(app, ["demo", "--expect", "comprar"])
    assert r.exit_code == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: `ModuleNotFoundError: No module named 'intent.cli'`

- [ ] **Step 3: Implement `cli.py``

`CliRunner.invoke` must not read the real environment in a way that changes
results, so the commands build their collaborators through module-level
factories that call `Settings.from_env()`. Tests rely on the ambient environment
being empty of `INTENT_*` variables; where that cannot be guaranteed, have
`from_env` read an explicit mapping the command passes down.

Exit codes per spec §9.8: 0 identified · 1 `desconhecido`/handoff · 2 invalid
input · 3 internal. Use `raise typer.Exit(code=...)`.

`demo` reads `tests/fixtures/pt_cases.json` when it exists and falls back to a
built-in copy of the same ten cases otherwise. It prints a table of expected
versus detected and exits 1 when any row disagrees. `--expect <intent>` treats
every case as expecting that intent, which is what makes the failure path
testable.

All human-readable output is PT-BR. `--json` output is machine-only and must be
the only thing on stdout in that mode, so it stays pipeable.

- [ ] **Step 4: Add the `rules-validate` target to the Makefile**

```makefile
rules-validate:
	$(PY) -m intent.cli rules validate src/intent/rules/pt.json
	$(PY) -m intent.cli rules validate src/intent/rules/en.json
```

- [ ] **Step 5: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: all passed

- [ ] **Step 6: Verify the entrypoints by hand**

```powershell
.\.venv\Scripts\python.exe -m intent.cli detect "quero cancelar meu plano"
.\.venv\Scripts\python.exe -m intent.cli demo
```

Expected: the first prints the intent and exits 0; the second prints the
expected-versus-detected table with all ten rows agreeing.

- [ ] **Step 7: Commit** (conditional)

```bash
git add src/intent/cli.py tests/test_cli.py Makefile
git commit -m "feat: Typer CLI with documented exit codes"
```

---

## Task 14: Web UI

**Files:**
- Create: `web/index.html`, `web/styles.css`, `web/js/api.js`, `web/js/format.js`,
  `web/js/intents.js`, `web/js/chat.js`, `web/js/app.js`
- Test: `tests/test_assets.py`

**Interfaces:**
- Consumes: the Task 12 HTTP API.
- Produces:
  - `api.js` — `detect(text, opts)`, `batch(texts)`, `listIntents(lang)`,
    `getRules(lang)`, `health()`, `chat(message, sessionId)`. Each returns
    `{ok, status, data}` and never throws on a non-2xx.
  - `format.js` — `formatConfidence(n)`, `formatScore(match)`,
    `explainUnsupported(language)`, `badgeClass(intent)`, `escapeText(s)`.
    **No imports** — `test_assets.py` evaluates this file directly, which is the
    only way to unit-test a pure JS function without adding Node to a Python
    project. Keep it that way, and say so in its header comment.
  - `intents.js` — `renderIntents(data)`, `renderTokens(tokens)`,
    `renderTrace(trace)`, `renderLanguage(language)`, `renderResult(result)`,
    `renderLog(entries)`, `accuracy(entries)`. Each takes a container element
    and builds DOM nodes; none assigns `innerHTML`.
  - `chat.js` — `mountChat(root)`, `send(message)`.
  - `app.js` — wires the controls, owns the run log capped at 20, and holds the
    `sessionId` from the last chat response.

- [ ] **Step 1: Write the failing tests**

```python
import pathlib
import re
import shutil
import subprocess

import pytest

WEB = pathlib.Path(__file__).parents[1] / "web"
JS = WEB / "js"
MODULES = ["api.js", "format.js", "intents.js", "chat.js", "app.js"]


def _all_assets():
    return list(WEB.rglob("*.js")) + list(WEB.rglob("*.html")) + list(WEB.rglob("*.css"))


def test_every_asset_exists():
    assert (WEB / "index.html").is_file()
    assert (WEB / "styles.css").is_file()
    for name in MODULES:
        assert (JS / name).is_file(), name


@pytest.mark.parametrize("name", MODULES)
def test_every_module_has_balanced_delimiters(name):
    # always-on structural check; test_every_module_parses is the real one
    source = (JS / name).read_text(encoding="utf-8")
    assert source.count("{") == source.count("}"), name
    assert source.count("(") == source.count(")"), name


@pytest.mark.parametrize("name", MODULES)
def test_every_module_parses(name):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    result = subprocess.run([node, "--check", str(JS / name)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("name", MODULES)
def test_no_module_uses_innerhtml(name):
    assert "innerHTML" not in (JS / name).read_text(encoding="utf-8"), name


def test_no_remote_reference_anywhere():
    for path in _all_assets():
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text, path
        assert "https://" not in text, path


def test_every_relative_import_resolves():
    for name in MODULES:
        source = (JS / name).read_text(encoding="utf-8")
        for target in re.findall(r"from\s+['\"](\./[^'\"]+)['\"]", source):
            assert (JS / target).is_file(), f"{name} -> {target}"


def test_format_js_stays_import_free():
    # test_format_helpers evaluates this file with exec(); an import breaks it
    source = (JS / "format.js").read_text(encoding="utf-8")
    assert "import" not in source


def test_index_loads_every_module_as_a_module():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'type="module"' in html
    assert 'src="./js/app.js"' in html
    for name in MODULES:
        assert name in html, name


def test_index_has_the_expected_controls():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for element_id in ("message", "expected-intent", "samples",
                       "require-supported", "detect", "clear"):
        assert f'id="{element_id}"' in html, element_id


def test_index_has_every_panel():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for panel in ("result", "explanation", "tokens", "trace", "language",
                  "chat", "log"):
        assert f'id="{panel}"' in html, panel


def test_every_control_has_a_label():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for element_id in ("message", "expected-intent", "samples"):
        assert f'for="{element_id}"' in html, element_id


def test_the_stylesheet_defines_badges_focus_and_reduced_motion():
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    assert ".badge" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css


def test_format_helpers():
    # deliberate: the only way to test pure JS from a Python suite
    body = (JS / "format.js").read_text(encoding="utf-8").replace("export ", "")
    namespace = {}
    exec(compile(body, "format.js", "exec"), namespace)
    assert namespace["formatConfidence"](0.7532) == "75%"
    assert namespace["formatConfidence"](0) == "0%"
    assert namespace["formatScore"]({"score": 3.5, "min_score": 3.0}) == "3.5 / 3.0"
    assert namespace["formatScore"]({"score": 0.0, "min_score": 3.0}) == "0.0 / 3.0"
    assert namespace["escapeText"]("<img onerror=x>") == "&lt;img onerror=x&gt;"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_assets.py -v`
Expected: failures — `web/` does not exist

- [ ] **Step 3: Write `web/index.html`**

Dark, terminal-flavoured, one page. Controls: `#message` textarea,
`#expected-intent` select, `#samples` select, `#require-supported` checkbox,
`#detect` and `#clear` buttons. Seven `<section>` panels with the ids
`test_index_has_every_panel` checks. Load `./js/app.js` as `type="module"`. Every
input gets a `<label for=...>`.

- [ ] **Step 4: Write `web/styles.css`**

CSS custom properties for the palette. A `.badge` variant per intent, each with
a `::after` text marker, so a badge never relies on colour alone. Include
`:focus-visible` outlines and a `@media (prefers-reduced-motion: reduce)` block
disabling transitions.

- [ ] **Step 5: Write the five modules**

`format.js` first, dependency-free, with the header comment explaining why it
must stay that way.

`api.js` uses `fetch` and returns `{ok, status, data}`. `intents.js` builds
every node with `document.createElement` and assigns only `textContent` and
`className`. The rule trace renders a negated hit with the text `NEGADO` and
`class="rule negated"`, and a hit that lost on score with `class="rule dropped"`.

`chat.js` renders turns and shows `pending_step` as "Passo N de 3". `app.js` wires
it together, keeps the `sessionId` from the last chat response, and caps the run
log at 20 with `accuracy(entries)` rendered into `#log`.

- [ ] **Step 6: Run to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_assets.py -v`
Expected: all passed, with skips only for `test_every_module_parses` if node is
absent

- [ ] **Step 7: Serve it and check by hand**

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:8000` and confirm: (a) the token table fills with
real lemmas; (b) `"não quero cancelar"` shows `desconhecido` with `NEGADO` on
the `cancelar` hit; (c) `"quero cancelar e depois assinar o plano Pro"` shows
two badges and `multi_intent`; (d) `<img onerror=alert(1)>` typed into the box
renders as visible text and does not fire.

- [ ] **Step 8: Commit** (conditional)

```bash
git add web tests/test_assets.py
git commit -m "feat: web UI with token, trace and expected-vs-detected panels"
```

---

## Task 15: Documentation, ADRs, and the final gate

**Files:**
- Create: `README.md`, `docs/adr/0001-rules-as-data.md`,
  `docs/adr/0002-lexicon-free-scoring.md`,
  `docs/adr/0003-abstain-on-unsupported-language.md`, `tests/test_docs.py`
- Modify: `.env.example` (verify completeness)

**Interfaces:**
- Consumes: the whole project.
- Produces: three ADRs, a README, and a test that keeps `.env.example` honest.

- [ ] **Step 1: Write the three ADRs**

Each short: Context, Decision, Consequences.

0001 records that rules are JSON content so the product team ships a change
without a deploy, and names the cost: manual maintenance and no automatic
generalisation. 0002 records that unlike the previous project in this folder
there is **no lexicon at all** — scoring comes only from rule weights, so there
is no polarity list to keep in sync and nothing to tune per word. 0003 records
that an unsupported language is reported rather than guessed, with the
`취소해줘` case as the reason.

- [ ] **Step 2: Write `README.md`**

Sections: what it is; requirements (Python 3.14, both spaCy models optional);
install (`make install`); run (`make serve`, `make test`, `make demo`,
`make rules-validate`); the rule file format with a worked `cancelar` example;
**how to add an intent** as a numbered procedure — add the id to both JSON
files, set `priority` and `min_score`, add rules, add examples, add a case to
`tests/fixtures/pt_cases.json`, run `make test`; the API table; the CLI table
with exit codes; and a note that `git` is unavailable in this environment, so
the commit steps were skipped.

- [ ] **Step 3: Write the failing test that keeps `.env.example` honest**

`tests/test_docs.py`:

```python
import dataclasses
import pathlib

from intent.config import Settings

ROOT = pathlib.Path(__file__).parents[1]
EXPECTED = {
    "INTENT_LANGUAGES", "INTENT_DB_PATH", "INTENT_MIN_TEXT_CHARS",
    "INTENT_MAX_TEXT_CHARS", "INTENT_NEGATION_WINDOW", "INTENT_MAX_INTENTS",
    "INTENT_MIN_CONFIDENCE", "INTENT_MIN_SCRIPT_SHARE",
    "INTENT_MIN_LANGUAGE_SHARE", "INTENT_MIN_LANGUAGE_MARGIN",
    "INTENT_MIXED_LANGUAGE_RATIO", "INTENT_BATCH_MAX_ITEMS", "INTENT_LOG_LEVEL",
}


def test_env_example_documents_every_setting():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in EXPECTED:
        assert name in text, name
    assert len(EXPECTED) == len(dataclasses.fields(Settings))


def test_env_example_has_no_setting_the_settings_class_lacks():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    import re
    documented = set(re.findall(r"INTENT_[A-Z_]+", text))
    assert documented == EXPECTED


def test_readme_documents_the_four_make_targets():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for target in ("make install", "make test", "make demo", "make serve"):
        assert target in text, target


def test_readme_explains_how_to_add_an_intent():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "como adicionar" in text or "add an intent" in text
```

- [ ] **Step 4: Run it, fix `.env.example` and the README until it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_docs.py -v`
Expected: passes once `.env.example` names all thirteen variables and the README
covers the four targets plus the add-an-intent procedure.

- [ ] **Step 5: Run the whole suite and both checks**

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m intent.cli demo
.\.venv\Scripts\python.exe -m intent.cli rules validate src/intent/rules/pt.json
.\.venv\Scripts\python.exe -m intent.cli rules validate src/intent/rules/en.json
```

Expected: every test passes; `demo` prints ten rows all agreeing and exits 0;
both rule files validate.

- [ ] **Step 6: Commit** (conditional)

```bash
git add README.md .env.example docs/adr tests/test_docs.py
git commit -m "docs: README, ADRs, and a verified env example"
```

---

## Appendix: dependency order

```
1  skeleton ─┬─ 2 normalise ─┬─ 3 language
             │               ├─ 4 rules ──────┐
             │               └─ 5 matching ───┼─ 6 negation ─ 7 ranking ─ 8 rule content
             │                                  │                            │
             └──────────────────────────────────┴────────────────────────────┴─ 9 pipeline
                                                                                    │
                                                            10 service ── 11 chat ─┤
                                                            12 api ── 13 cli ──────┤
                                                            14 ui (needs 12) ───────┤
                                                            15 docs (needs all) ───┘
```

Tasks 3, 4 and 5 depend only on Task 2 and are independent of each other, so
they are the only work that can be dispatched in parallel. Everything from Task 6
onward is strictly sequential, because each task's fixtures encode the previous
task's scoring decisions. Even for the parallel three, each still needs its own
review: they disagree about what a rule *means*, which is exactly the kind of
disagreement a single reviewer will not catch.

# Intent Detection Engine — Design Spec

**Status:** approved (verbal approval, 2026-09-26)
**Stack:** Python 3.14 · spaCy 3.8 · FastAPI · Typer · vanilla ES modules
**Language of docs/UI:** PT-BR · **Language of code identifiers:** English
**Working dir:** `C:\Users\Aluno\Downloads\Nova pasta\aa`

---

## 1. Problem

A chatbot must decide what the user is trying to do from a single free-text
message. Guessing wrong is expensive: misreading "quero cancelar" as a support
question routes the user to a human instead of ending a subscription.

This engine answers exactly one question: **which intent, if any, is expressed
in this message, and how sure are we?** It is a deterministic rules engine, not
a statistical classifier. Same input, same output, always.

Intents to recognise, at minimum: `cancelar`, `comprar`, `suporte`.

---

## 2. Scope

### In scope

- Versioned, declarative rule files edited by non-engineers.
- Four rule types: `keyword`, `lemma`, `pattern`, `all`.
- Priority-based disambiguation and multi-intent detection.
- Negation, so "não quero cancelar" never yields `cancelar`.
- Explicit language handling with abstention on unsupported input.
- Library core, REST API, CLI, and a visual web UI.
- A rules-based chatbot that consumes the engine, so intent → action is
  demonstrable end to end.

### Out of scope

- Statistical/ML classification, training corpora, embeddings.
- Slot filling beyond the chat collector's fixed ordered questions.
- Deployment, accounts, billing, real ticket systems.
- Vector stores, agents, tool calling, LLM calls. No network at runtime.

### Non-goal worth naming

"did the engine work?" is measured by comparing the detected intent against the
intent the user *expected*. The UI therefore lets you declare the expected
intent before running a sample.

---

## 3. Architecture

Same five-layer structure as the previous project in this folder, so the
familiarity is free.

```
web/                      static UI, served by FastAPI, no build step
api.py                    HTTP layer, Pydantic models
cli.py                    Typer commands
service.py                application use cases, orchestration only
normalize.py matching.py  the engine: normalisation/index map, rules, scoring
pipeline.py               spaCy loading, orchestration, TokenView assembly
language.py rules.py      language gating, rule loading/validation
config.py errors.py       settings, typed error codes
schemas.py                value objects
```

`service.py` holds no scoring logic. It delegates to the pipeline, persists, and
shapes the response.

### Dependency direction

```
cli/api  ->  service  ->  pipeline  ->  matching  ->  rules
                   |            \->  normalize
                   \->  language
```

`rules.py`, `normalize.py` and `language.py` know nothing about the engine.
`matching.py` never touches the filesystem or settings. Everything is
injectable, so tests need no `monkeypatch`.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` from env, all defaults, `resources_dir()` |
| `errors.py` | `IntentError(code, message, details)` and subclasses |
| `schemas.py` | `IntentMatch`, `IntentResult`, `LanguageInfo`, `TokenView`, `RuleView` |
| `normalize.py` | `normalise()`, the accent-stripped copy, and the offset index map |
| `rules.py` | Load/validate `rules/{lang}.json`, `RuleSet`, `IntentSpec` |
| `language.py` | Script + function-word language identification |
| `pipeline.py` | Normalise, identify language, build spaCy doc, call matcher, assemble `IntentResult` |
| `matching.py` | Evaluate rules, negation, aggregation, ranking |
| `chat.py` | `Chatbot` — intent → scripted response, session state, PII redaction |
| `service.py` | Use cases: detect, batch, catalogue, chat, health, history |
| `api.py` | Routes, error envelope, static files |
| `cli.py` | Typer app |
| `web/` | `index.html`, `styles.css`, `js/{api,format,intents,app,chat}.js` |

`normalize.py` is split out from `pipeline.py` because the index map is one
subtle invariant with its own failure mode, and burying it in the orchestrator
is how offset bugs survive.

---

## 4. Rules are data

Rules live in `src/intent/rules/{lang}.json`. They are content, not code: the
product team edits them, bumps `version`, and deploys. The engine never has
rule-specific branches.

```json
{
  "language": "pt",
  "version": "1.0.0",
  "intents": [
    {
      "id": "cancelar",
      "label": "Cancelar assinatura",
      "priority": 90,
      "min_score": 3.0,
      "rules": [
        { "type": "keyword", "value": "cancelar", "weight": 2.0 },
        { "type": "lemma", "value": "cancelamento", "weight": 2.0 },
        { "type": "pattern", "value": "cancelar\\s+(meu\\s+)?(plano|assinatura|conta)", "weight": 3.0 }
      ],
      "examples": ["quero cancelar meu plano", "como faço para cancelar a assinatura?"]
    },
    {
      "id": "suporte",
      "label": "Suporte técnico",
      "priority": 60,
      "min_score": 2.5,
      "rules": [
        { "type": "pattern", "value": "n(ã|a)o\\s+(funciona|abre|carrega|responde)", "weight": 2.5 },
        { "type": "keyword", "value": "suporte", "weight": 2.0 },
        { "type": "lemma", "value": "erro", "weight": 1.0 }
      ],
      "examples": ["a página não carrega", "preciso de suporte"]
    }
  ]
}
```

### 4.1 Rule types

| `type` | Matches on | Engine mechanism | Example |
|---|---|---|---|
| `keyword` | exact surface form of a token | `token.lower_ == value.lower()` | `cancelar` |
| `lemma` | lemmatised form | `token.lemma_.lower() == value.lower()` | `cancelamento` |
| `pattern` | regex over accent-stripped, lowercased text | `re.search(value, normalised)` | `cancelar\s+(meu\s+)?plano` |
| `all` | every string present as a token | conjunction of `keyword`/`lemma` checks | `["cancelar", "assinatura"]` |

`all` values are **single tokens**. A value containing whitespace is rejected at
load. The type exists to demand *two independent signals*, and a multi-word
value is a phrase, which is what `pattern` is for. Keeping `all` to single
tokens also keeps token consumption well-defined: an `all` rule claims whole
tokens, so there is no partial overlap to reason about.

`pattern` compiles once at load time, wrapped in implicit word boundaries
`(?<!\w)…(?!\w)` so a rule cannot fire inside a longer word. Without this,
`pattern: "assinar"` matches inside `desassinar`, and a user saying they *un*
subscribed would trigger `comprar`. The implicit boundaries make partial-word
matching impossible by default; a pattern that genuinely needs to match a
prefix must state it, which is the right place for that decision to be visible.
Invalid regex is a `RulesError` at load, not a crash on the first user message.

A `pattern` match is mapped back to the token span that contains its start
offset, so a regex hit still appears in the token table at a real position.
A regex that starts mid-token is reported with the token it overlaps.

### 4.2 Why `pattern` is accent-stripped

The user can type anything. Normalisation lowercases and strips combining
accents (`unicodedata` NFD) for regex matching, so `nao` and `não` match the
same rule. `keyword`/`lemma` still compare against the original forms, and
matchers try both the accented and stripped form of a rule value.

`pattern` values are normalised at load time by the same function, so a rule
written as `não\s+funciona` still matches. A rule author never has to think
about accents for regex.

### 4.3 Why `all` exists

`keyword: "cancelar"` alone fires on "quero cancelar meu plano" (score 2.0),
but `suporte`'s `lemma: erro` alone is a weak signal. `all` lets a rule demand
two independent pieces of evidence, which is how weak signals are kept from
winning.

Tokens matched by an `all` rule are **consumed** for that intent: no other rule
of that intent, of any type, may count them again. Without this,
`all: ["cancelar", "assinatura"]` plus `keyword: "cancelar"` would score 5.5 for
one mention of the word.

Consumption is applied by evaluating every `all` rule of an intent **before** any
other rule, not by array position. A single pass would let `keyword cancelar`
score 2.0 first and then let `all` claim the same token anyway, so the total
would depend on how the JSON happened to be written. Rules files are content,
and content order must not change a score.

### 4.4 `min_score` is the false-positive guard

An intent is reported only if its total score is at least its own `min_score`.
`cancelar` sets it high (3.0) because that intent is the expensive one to get
wrong: a false `cancelar` starts subscription-cancellation on a support
question. `saudacao` sets it low (1.0) because greeting detection should be
eager. The number belongs next to the rule that earns it.

There is no separate global rule-file threshold. A single global gate exists in
settings as `INTENT_MIN_CONFIDENCE` (default 0.3), acting on the winning intent
only. Two knobs with two distinct jobs: `min_score` is per-intent content
policy written by whoever owns the rule; `min_confidence` is a system-level
floor that exists to catch a rules file whose `min_score` was mis-set. With
correct rules the floor never fires independently — that is expected, and 5.5
says so plainly rather than implying two independent gates.

### 4.5 Validation at load

Rejected with `RulesError` and a precise path:

- unknown `type`
- `keyword`/`lemma` with a value containing whitespace
- `pattern` that is not a valid regex
- `all` with fewer than 2 strings, or with a string containing whitespace
- non-positive or non-finite `weight`
- `min_score` ≤ 0
- duplicate intent `id`
- duplicate `type`+`value` within one intent, which would double-count
- `id` not matching `^[a-z][a-z0-9_]*$`
- `version` not semantic (`MAJOR.MINOR.PATCH`)

A JSON integer `weight` such as `2` is **accepted** and coerced to `2.0`. These
files are hand-edited by people who are not writing Python, and rejecting
`"weight": 2` because it lacks a decimal point would be a hostile way to spend
their afternoon. Coercion happens at load; only genuinely unusable values fail.

The engine is strict on load and silent at runtime.

---

## 5. Detection pipeline

```
normalise text
  -> identify language
  -> early gates (empty, too short, too long, unsupported)
  -> spaCy Doc via nlp.pipe
  -> match rules (keyword / lemma / pattern / all)
  -> apply negation
  -> aggregate score per intent
  -> apply per-intent min_score
  -> rank by (score desc, priority desc, id asc)
  -> confidence + INTENT_MIN_CONFIDENCE gate
  -> IntentResult
```

### 5.1 Normalisation

Keep the original text for token views. For matching, build a normalised copy:
lowercase, NFD, strip combining marks, collapse runs of whitespace, keep
sentence-ending punctuation so `pattern` can rely on `.?!` boundaries.

**Length gates, before anything else.** Text shorter than
`INTENT_MIN_TEXT_CHARS` (2) or longer than `INTENT_MAX_TEXT_CHARS` (5000) is
rejected as `input_invalid`: `422` from HTTP, exit code `2` from the CLI. It is
a client error, not an abstention, and reporting it as `desconhecido` would
tell the user their message was understood when it never reached the rules.

**No usable text is not a short message.** A string can be long enough to pass
the length gate and still contain nothing to reason about: `"!!!"`, `"   "`,
`"..."`, `"12345"`. If the normalised text has no cased character, the result is
`empty_text` / `supported: false` (6.2), not `insufficient_signal` and not a
crash in the profile counter.

**Normalisation is lossy, so offsets are mapped, not assumed.** Stripping
`ã` (`a` + combining tilde) removes a character, so `"não funciona"` normalises
to `"nao funciona"` and every offset after the accented character shifts. A
regex match at normalised index 11 is original index 12. `normalise()` therefore
returns the normalised string **together with an index map**, one entry per
normalised character, holding the original index it came from. Every reported
`char_start`/`char_end` is translated through that map, so the UI can highlight
the exact original text the rule fired on.

When a character has no counterpart — an inserted or removed combining mark at
the map boundary — the translation clamps to the nearest mapped index, and the
trace entry carries `"offsets_exact": false` rather than a silently wrong
span. For non-Latin scripts, where decomposition does not apply, the map is the
identity and `LanguageInfo.offset_preserved` is `true`; the flag is only `false`
when normalisation genuinely cannot be aligned.

### 5.2 Negation

A match is discarded if a negator appears in the window of
`INTENT_NEGATION_WINDOW` tokens (default 3) to its left. A negated match is
**kept in the trace** with `"negated": true` and zeroed weight, so the UI can
show that the engine saw the phrase and understood it. This is the difference
between "my engine failed" and "my engine decided correctly", and it is the
single most valuable diagnostic in the whole UI.

Scan backwards one token at a time from the match's start token. At each
position, **check the negator first, then the clause boundary**. A boundary stops
the scan. Stop the scan on `.` `,` `;` `:` `!` `?` and a newline, and on the
clause words `mas` `porém` `porque` `pois` `então` `e` — so a negation in one
clause does not reach into the next, including when the user pasted two
messages separated by a line break.

| Message | Negated | Result |
|---|---|---|
| `não quero cancelar` | `cancelar` | `desconhecido` |
| `quero cancelar e não comprar` | `comprar` | `cancelar` only |
| `não gostei, mas quero cancelar` | — | `cancelar` (boundary stops the scan) |
| `deixar de cancelar` | `cancelar` | `desconhecido` (multi-word negator) |
| `não quero cancelar\ne quero comprar` | `cancelar` only | `comprar` (newline is a boundary) |

Negators are phrases, not just words, so `deixar de` is matched as a unit and
`deixar`/`de` alone is not in the list. A message that is *only* a negator
(`"não"`) has nothing to negate and must not error on the way to `desconhecido`.

### 5.3 Scoring

- `score(intent) = Σ weight` of non-negated rules that matched, where **each
  rule contributes its weight at most once per intent**. `"cancelar cancelar
  cancelar"` scores exactly what `"cancelar"` scores. Every occurrence is still
  recorded in the trace, so repetition is visible without being rewarded.
- `confidence = score / (score + 1)`, saturating at 1.0 but never reaching it
  from a single rule.
- A trace entry is emitted for every rule occurrence that matched, including
  negated ones, with `weight`, `char_start`, `char_end`, and `negated`.

Counting a rule once is the cheapest honest defence against score inflation:
without it, four repetitions of a 2.0 keyword score 8.0 and a user typing
angrily crosses `cancelar`'s 3.0 threshold twice over. Adding a repetition
dampening parameter instead would be a second knob doing the job of a
deduplication that should have existed anyway.

Ranking is `(score desc, priority desc, id asc)`. Total and deterministic, so
every ambiguous case is a fixed test rather than a flaky one.

### 5.4 Multi-intent

All intents above `min_score` are reported, capped at `INTENT_MAX_INTENTS`
(default 3), and `multi_intent` is `true` when more than one survives.
`primary` is the first after ranking. "quero cancelar e assinar o plano Pro"
yields `[cancelar, comprar]` — collapsing that to one intent would drop half
the user's request.

### 5.5 `desconhecido` and the confidence gate

If the winning intent's confidence is below `INTENT_MIN_CONFIDENCE` (default
0.3) the engine reports no intent: `primary: "desconhecido"`, `intents: []`.
An uncertain match is worse than an honest miss, because the chatbot will act
on it. All signals remain in the trace, so the UI still shows the near-misses
and their scores.

Note what this gate is and is not. `min_score` is the per-intent content policy
and does the real filtering. The confidence gate is a single system-level floor
whose only job is to catch a rules file whose `min_score` was set too low during
a content edit. With the shipped rules it never fires on its own, and pretending
otherwise would imply a second layer of filtering that does not exist.

---

## 6. Language handling

Same reasoning as the previous project: a rule set only covers the languages it
was written for, so a message in an unsupported language must be reported as
unsupported, not guessed at. "취소해줘" must not land in `desconhecido` and then
be filed as a cancellation ticket.

### 6.1 Two-stage detection

**Stage 1 — script.** Classify each character by `unicodedata` name prefix and
take the share of cased characters per script family: Latin, CJK (Han/Hangul/
Kana), Cyrillic, Arabic, else unknown. If the Latin share is below
`INTENT_MIN_SCRIPT_SHARE` (0.85) the message is not predominantly Latin script,
and the answer is immediately unsupported. There is no point guessing from word
frequencies there, because no rule file exists for those families.

The 0.85 threshold is what lets a message that is 97% Latin with a stray emoji
or CJK symbol through to stage 2, while a message that is genuinely Japanese
stops here. Deciding this by the leading family instead would be worse: a text
that is 40% Latin and 60% Cyrillic has a Latin-leading *minority*, and routing
it to stage 2 would produce a confident wrong answer from three borrowed words.

**Stage 2 — function-word profiles.** Within Latin script, count stopwords from
`resources/language_profiles.json` per language and pick the highest share.
`INTENT_MIN_LANGUAGE_SHARE` (0.55) and `INTENT_MIN_LANGUAGE_MARGIN` (0.15) guard
the decision:

- top share below the floor → `insufficient_signal`
- gap to second place below the margin → `ambiguous_language`
- `INTENT_MIXED_LANGUAGE_RATIO` (0.25): a language holding less than this
  fraction of the decision tokens is ignored, so one English word in a long
  Portuguese message does not trigger ambiguity

`Resources` are shared (digits, punctuation, currency — not evidence) and
exclusive (stopwords — evidence).

### 6.2 Outcome

| Case | `supported` | Reason |
|---|---|---|
| Clear PT or EN | `true` | `ok` |
| Latin, shares too low | `false` | `insufficient_signal` |
| Latin, top two too close | `false` | `ambiguous_language` |
| Below 85% Latin script (CJK, Cyrillic, Arabic, or heavy mixing) | `false` | `unsupported_script` |
| No usable text | `false` | `empty_text` |

### 6.3 Models

Load the real model when installed, else fall back to
`spacy.blank(language)` + `sentencizer`. `lemma` rules are skipped when the
active pipeline cannot lemmatise, and `ResourceReport` lists them under
`limitations` rather than failing the request. Every result carries the
`ResourceReport`, so the UI never shows a confident answer it cannot justify.

The three required behaviours are covered by three different mechanisms —
`empty_text` gate, `insufficient_signal` (Latin), `unsupported_script` (CJK) —
so a test of one cannot pass by accident for another.

---

## 7. Intent catalogue

Ten business intents plus one rule-driven system intent. `label` and
`examples` are PT-BR; ids are English.

| id | priority | min_score | Intent |
|---|---|---|---|
| `cancelar` | 90 | 3.0 | Cancelar assinatura / encerrar serviço |
| `reclamacao` | 80 | 3.0 | Reclamação formal, exigir escalonamento |
| `pagamento` | 75 | 2.5 | Cobrança, boleto, pix, cartão, estorno, reembolso |
| `acesso` | 70 | 2.5 | Login, senha, 2FA, conta bloqueada |
| `upgrade` | 65 | 3.0 | Mudar para plano maior |
| `comprar` | 60 | 2.0 | Contratar / assinar novo plano |
| `downgrade` | 55 | 3.0 | Mudar para plano menor |
| `suporte` | 50 | 2.5 | Erro, página que não abre, dúvida técnica |
| `alterar_dados` | 45 | 3.0 | Nome, e-mail, endereço, documentos |
| `duvida` | 40 | 2.5 | Pergunta geral sobre o produto |
| `saudacao` | 20 | 1.0 | System, rule-driven. Saudações, "oi", "bom dia" |

`desconhecido` is the other system intent, and it is deliberately absent from
the table: it has no rules and never competes in ranking. It is the label the
engine emits when nothing clears `min_score` (5.5) or the language is
unsupported. An intent that had to compete for a score could never reliably mean
"no intent at all".

### 7.1 Catalogue is per language

`intents` is the PT-BR catalogue. The EN file carries the same eleven
rule-driven ids so the two catalogues stay comparable in the UI's expected-vs-
detected table; a handful of id/score differences are acceptable. If a language
file omits an id, that intent is simply unavailable in that language, and
`GET /api/v1/intents?language=en` reports it as absent rather than scoring it
as zero against a rule that does not exist.

### 7.1 Why `atendimento` is excluded

With enough rules, almost every angry message also contains a support word, so
an `atendimento` intent scores high on nearly every complaint. It would win ties
against `reclamacao` and `suporte` and destroy the metric. The intent exists
upstream of this engine: unknown or escalating intents become a human handoff,
which `desconhecido` already covers.

### 7.2 Priority rationale

`cancelar` is highest because missing it keeps a customer subscribed and paying.
`reclamacao` is next because it demands a human. `saudacao` is lowest so a
greeting never displaces a real request. Priority is the tie-break only; score
leads.

---

## 8. `require_supported`

The client chooses what an unsupported language means.

- Default: `200` with `label: "unsupported"`, `reason`, and the language info.
- `require_supported: true`: `422 unsupported_language`, same body plus
  `detected_language`, so the caller can offer a choice.

`label` is exactly `"unsupported"`, never an intent id. The UI has a dedicated
unsupported state, and `POST /api/v1/chat` falls back to a handoff message in
the user's own language.

**This flag is about language, never about confidence.** `require_supported:
true` turns an unsupported *language* into a 422 and nothing else. A Portuguese
message that yields no intent still returns `200` with
`primary: "desconhecido"`, because the language was perfectly identifiable and
the client's actual requirement — "only accept Portuguese" — is satisfied.
Folding low confidence into the same 422 would make a well-formed request look
malformed, and a client that retries on 422 would loop forever.

---

## 9. Public contracts

### 9.1 `IntentMatch`

`intent` · `label` · `score` (rounded to 4) · `confidence` (rounded to 4) ·
`priority` · `rule_hits[]` (`rule_id`, `type`, `value`, `weight`, `char_start`,
`char_end`, `negated`, `matched_text`)

### 9.2 `IntentResult`

`text` (original) · `normalised` · `intents[]` · `primary` · `multi_intent` ·
`confidence` (top) · `tokens[]` (`TokenView`) · `language` (`LanguageInfo`) ·
`explanation` (PT-BR) · `rules_version` · `trace[]` (every match, negated
included) · `resources` (`ResourceReport`) · `duration_ms`

### 9.3 `LanguageInfo`

`language` · `supported` · `reason` (`ok` | `insufficient_signal` |
`ambiguous_language` | `unsupported_script` | `empty_text`) ·
`script` · `offset_preserved` · `candidates[]` (`language`, `share`, `markers`,
exclusive vs shared) · `min_share` and `margin` when the decision is rejected,
plus the two candidates that tied, so a rejected verdict can be diagnosed
instead of merely observed

### 9.4 `ResourceReport`

`models` (`language` → `model`, `fallback`) · `rules` (`language` →
`version`, `entries`, `sha256`) · `limitations[]` · `warnings[]`

The rules digest, not a lexicon digest: this project has no lexicon, and the
value of the hash is letting an operator prove which rule file produced a given
result after someone edits the JSON.

### 9.5 `TokenView`

`text` · `lemma` · `pos` · `is_stop` · `is_sentence_start` · `matched` (a rule
fired on this token) · `negated` (it fired but was discarded by negation)

`matched`/`negated` are what make the token table worth reading. Without them
the panel shows generic linguistic annotation that tells you nothing about the
decision; with them the decisive tokens are visibly marked, including the ones
the engine correctly threw away. There is deliberately no `sentiment` field —
`Doc.similarity` needs word vectors, which the small models used here do not
carry, so such a field would always be empty or misleading.

The trailing-underscore spaCy attributes (`token.lemma_`, `token.pos_`,
`token.is_stop`) stay inside `pipeline.py`. Everything crossing a module
boundary uses clean names.

### 9.6 HTTP

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/api/v1/detect` | One message. `422 input_invalid` on bad body, `422 unsupported_language` when required |
| `POST` | `/api/v1/detect/batch` | ≤ 500, grouped by language, `nlp.pipe` per group |
| `GET` | `/api/v1/intents` | Catalogue with per-intent rule counts |
| `GET` | `/api/v1/rules/{lang}` | Active rules; `404 unknown_language` |
| `POST` | `/api/v1/rules/validate` | Validate a rules document without activating it |
| `GET` | `/api/v1/health` | Models in use, fallbacks, limitations, `rules_versions` |
| `GET` | `/api/v1/conversations` | Stored conversation summaries |
| `POST` | `/api/v1/chat` | `Chatbot.responder`, returning the reply plus the intents that drove it |
| `GET` | `/docs`, `/openapi.json` | FastAPI docs |

`POST /api/v1/chat` request: `message`, optional `session_id`, optional
`require_supported`, optional `language_hint`. Response: `reply`, `reply_format`,
`session_id`, `intents`, `primary`, `multi_intent`, `strategy`, `confidence`,
`turn_index`, `pending_step`.

### 9.7 Error codes

`input_invalid` · `unsupported_language` · `unknown_language` · `unknown_intent`
· `invalid_rules` · `model_unavailable` · `storage_error` · `internal`

`Errors` codes map 1:1 to exception classes so the envelope cannot drift.
`retryable: true` only for `model_unavailable` and `storage_error`.

### 9.8 CLI

| Command | Purpose |
|---|---|
| `intent detect "..."` | Detect, pretty or `--json` |
| `intent detect --stdin` | One message per line |
| `intent batch file.json` | Array of messages |
| `intent list` | Catalogue with rule counts |
| `intent rules show <lang>` | Active rules |
| `intent rules validate <file>` | Validate without activating |
| `intent demo` | Built-in examples, per intent |
| `intent chat [--session <id>]` | Interactive loop |
| `intent serve` | Uvicorn, `reload` by default |

Exit codes: `0` intent identified · `1` `desconhecido`/handoff · `2` invalid
input · `3` internal error. Tests assert them.

---

## 10. The chatbot

`Chatbot.responder(mensagem, session_id=None) -> ChatResponse`. Rule-based, one
strategy per intent. No generation.

| Intent | Strategy |
|---|---|
| `saudacao` | Greeting plus a short menu |
| `cancelar` | 3-step collector: confirm → reason → protocol. Advances one step per turn |
| `comprar` | Show plan catalogue, ask which plan |
| `upgrade` | Same, mentioning the current plan |
| `downgrade` | Same, mentioning the current plan |
| `suporte` | Triage; a support signal with an error mention issues `SUP-####` |
| `acesso` | 3-step recovery. **Never** asks the user for a password |
| `pagamento` | Mentions "cobrou duas vezes" → refund flow; else payment-method guidance |
| `alterar_dados` | Confirm which field, then acknowledge |
| `reclamacao` | Acknowledge, register a ticket, offer handoff |
| `duvida` | Answer from a small canned knowledge list, else handoff |
| `desconhecido` | Handoff to a human, quoting the original message |

Session state: `session_id`, `turns[]`, `pending_intent`, `pending_step`,
`collected{}`. Persisted to SQLite, which stores conversations only — the UI's
run log is client-side, so there is one store and not two.

**Routing order.** A pending collector takes precedence over fresh detection,
because a collector's follow-up turn ("sim", "não", "o plano Pro") almost
never contains a recognisable intent. Each turn is routed in this order:

1. Detect the intent as normal.
2. If `pending_intent` is set **and** the detected intent is `desconhecido`,
   or scores below its own `min_score`, advance the collector.
3. If a *different* intent clears `min_score`, abandon the collector and answer
   the new intent. A user who says "cancelar" and then "quero comprar um
   plano" has changed their mind, and the bot must follow them there.
4. `desconhecido` with no pending collector is a handoff.

Step 3 is the one that needs a test. A collector that swallows every subsequent
turn is a worse bug than a missing feature, because it silently discards what
the user actually asked.

Multi-intent: the bot answers `primary` and explicitly mentions the secondary,
so "quero cancelar e assinar o Pro" produces one coherent reply instead of two
disconnected ones.

**PII.** `SENSITIVE_PATTERNS` (card-like digits, "minha senha é", bearer tokens)
are redacted from the transcript before anything is stored or echoed, and the
user is told not to share them. The bot never requests a password or full card
number. This is a rule with a test, not a disclaimer.

---

## 11. Configuration

All settings from env, prefix `INTENT_`, defaults work with no env at all.

| Variable | Default |
|---|---|
| `INTENT_LANGUAGES` | `pt,en` |
| `INTENT_DB_PATH` | `""` (disabled) |
| `INTENT_MIN_TEXT_CHARS` | `2` |
| `INTENT_MAX_TEXT_CHARS` | `5000` |
| `INTENT_NEGATION_WINDOW` | `3` |
| `INTENT_MAX_INTENTS` | `3` |
| `INTENT_MIN_CONFIDENCE` | `0.3` |
| `INTENT_MIN_SCRIPT_SHARE` | `0.85` |
| `INTENT_MIN_LANGUAGE_SHARE` | `0.55` |
| `INTENT_MIN_LANGUAGE_MARGIN` | `0.15` |
| `INTENT_MIXED_LANGUAGE_RATIO` | `0.25` |
| `INTENT_BATCH_MAX_ITEMS` | `500` |
| `INTENT_LOG_LEVEL` | `INFO` |

Every variable here is read by named code in this spec. A setting nothing
consumes is a setting that will be tuned by accident and trusted by nobody, so
none are listed speculatively.

Models are optional. `make install` downloads `pt_core_news_sm` and
`en_core_web_sm`; without them the engine still works through the blank-pipe
fallback, and `lemma` rules are skipped and reported under `limitations`.

---

## 12. Visual UI

`web/`, ES modules, no framework, no build, no CDN. One page with a dark
terminal-style surface.

Controls: free-text input, **expected-intent selector** (the demo's ground
truth), a samples dropdown covering every intent, `require_supported`
checkbox, and Detect / Clear buttons. Enter submits.

Panels:

1. **Result** — primary intent badge, secondary intents, `multi_intent` flag,
   confidence bar, and `score / min_score` so an intent near its threshold is
   visibly marginal.
2. **Explanation** — the PT-BR sentence naming which rules fired.
3. **Token pipeline** — table of `text`, `lemma`, `pos`, `is_stop`, `matched`,
   `negated` from the actual spaCy `Doc`, with the decisive tokens marked and
   the model in use named. This is the spaCy demo.
4. **Rule trace** — every matched rule with type, value, weight, offsets, and a
   visible `NEGADO` marker. Failing rules appear struck through.
5. **Language** — script, candidates with shares, and the rejection reason.
6. **Chat** — turn-by-turn, bot bubbles, the pending collector step.
7. **Session log** — the last 20 runs with the expected-vs-detected pair, and
   an accuracy readout. This is the demo's payoff.

Rules for the UI: a badge never conveys meaning by colour alone, the contrast
ratio clears WCAG AA, keyboard navigation is complete, and the page works with
`prefers-reduced-motion`. No `innerHTML` with message data; `textContent` and
DOM construction only, so a message like `<img onerror=...>` renders as text.

---

## 13. Testing

`pytest`; every test builds its dependencies through constructors. No
`monkeypatch` of module attributes anywhere in the suite.

**Rule engine** — each of the four types in isolation; `pattern` across
punctuation; invalid regex rejected at load; `all` requiring both strings;
`all` tokens consumed so a following `keyword` cannot double-count them;
`min_score` below threshold drops the intent.

**Normalisation** — accents stripped for matching but the reported span still
lands on the original text, asserted with an accented message
(`"não consigo cancelar"`, where a naive offset would be off by two). A rule
written with an accent matches an unaccented message. `offsets_exact` is `false`
rather than wrong at an unmappable boundary.

**Negation** — all four rows of the table in 5.2, plus a negator immediately
before the match and a negator beyond the window.

**Language** — pt, en, mixed, CJK, Latin gibberish, empty, digits only. Assert
`reason` and `supported` separately, since the interesting failure is a wrong
reason, not a wrong boolean.

**Ranking** — two intents tied on score resolved by priority; equal priority
resolved by id; `multi_intent` cap; confidence gate.

**Fixtures** — five cases, because these are where rule engines actually rot:

1. `"quero sair!!!"` — needs a rule that does not exist yet, and the trailing
   `!!!` proves punctuation does not block matching. Forces a real rule addition
   rather than a clever test.
2. `"não quero cancelar, quero saber do reembolso"` — must yield
   `pagamento`, never `cancelar`. Two intents of opposite polarity, and the one
   place a naive keyword engine produces a genuinely expensive error.
3. `"o problema é que não consigo acessr"` — a typo. spaCy's lemmatiser will not
   repair `acessr`; it is not a known form. The asserted result is
   `desconhecido`, not `acesso`. This fixture and the next one are the pair
   that holds §15 honest: fuzzy matching is refused on purpose.
4. `"quero cancelar mas mudei de ideia, na verdade quero um plano maior"` —
   must yield `upgrade`. Here `cancelar` is **not** negated: the reversal words
   come after it, and negation only ever looks left. `upgrade` has to win on
   score, which means the rules file must carry an explicit reversal phrase at a
   weight high enough to outweigh three separate cancellation signals. This
   fixture is what forces that rule to exist, and it is the honest reason
   reversal phrases get a weight no ordinary phrase would need.
5. `"quero cancelar e depois assinar o plano Pro"` — `[cancelar, comprar]` with
   `multi_intent: true`, and a single coherent chat reply covering both.

**API** — status codes and error envelopes; `require_supported`; the empty-body
422; batch with a mixed-language payload; static file serving.
**CLI** — every exit code.
**UI assets** — every module parses, every import resolves, no CDN reference.
**Chat** — one turn per intent, multi-intent reply, PII redaction, collector
advancing across turns, and the case that matters most: a pending `cancelar`
collector abandoned mid-flow when the user asks for an upgrade instead (routing
step 3 in §10). A collector that swallows every later turn is a silent data-loss
bug, so it gets its own test.

Review focus: the negation window boundary, the confidence gate, and
`min_score` calibration for `cancelar`. Those are the three places where a
change silently alters behaviour everywhere.

---

## 14. Structure

```
aa/
  pyproject.toml            setuptools, src layout, pt/en extras
  README.md  Makefile  .env.example  .gitignore
  app.py                    uvicorn entrypoint (delegates to api:app)
  src/intent/
    __init__.py config.py errors.py schemas.py normalize.py
    rules.py language.py matching.py pipeline.py chat.py
    service.py api.py cli.py
    rules/pt.json  rules/en.json
  resources/language_profiles.json
  web/index.html  web/styles.css
  web/js/{api,format,intents,app,chat}.js
  tests/
    conftest.py  fixtures/
    test_normalisation.py  test_rules.py  test_language.py
    test_matching.py  test_pipeline.py  test_negation.py  test_service.py
    test_chat.py  test_api.py  test_cli.py  test_assets.py
  data/                     runtime, gitignored
  docs/adr/0001-rules-as-data.md
  docs/adr/0002-lexicon-free-scoring.md
  docs/adr/0003-abstain-on-unsupported-language.md
  docs/superpowers/specs/…    docs/superpowers/plans/…
```

`test_normalisation.py` exists as its own file because the index map has a
single responsibility and one subtle invariant. Buried in `test_pipeline.py` it
would be the first thing to stop being re-run when the pipeline changes.

The three ADRs record the decisions that look wrong until you have lived with
them: rules as data, no lexicon at all (unlike the previous project), and
abstention over guessing.

---

## 15. Accepted trade-offs

- **Rules do not generalise.** A phrasing nobody wrote a rule for is
  `desconhecido`, not a wrong guess. For a router that gates expensive
  actions, that is the right failure.
- **No fuzzy matching.** Deliberate. `similarity`/`fuzzywuzzy` are rejected
  because a near-miss on a cancellation rule cancels a real subscription.
- **Rule maintenance is manual.** The cost of explicit rules is editing them.
  Cheaper than the cost of the false `cancelar`.
- **Regex in JSON is powerful and unguarded.** Mitigated by strict validation
  at load, not by restricting what rules may contain.
- **Only pt and en.** Two rule files is already a maintenance commitment;
  a third language multiplies it and every new language needs its own
  functions-word profile.
- **`desconhecido` sends users to a human.** A bot that never escalates
  becomes the bottleneck it replaced.
- **No authentication.** The API is local and read-mostly; adding auth would
  be scope the task never asked for.
- **No production deployment.** Explicitly out of scope.

---

## 16. Done means

- `make test` green; the negation table, ranking, and all five fixtures
  covered.
- `make demo` prints the fixture cases with expected vs detected intent, and
  `acessr` correctly reports `desconhecido` rather than `acesso`.
- `make serve` serves the UI, and the token panel shows real spaCy tokens with
  the matched and negated ones marked.
- The unsupported-language case abstains with a reason, on a message that is
  clearly not Portuguese or English.
- `make rules-validate` accepts the two shipped rule files and rejects a
  deliberately broken one.
- `.env.example` documents every variable; `README.md` covers install, usage,
  the rule format, and how to add an intent.

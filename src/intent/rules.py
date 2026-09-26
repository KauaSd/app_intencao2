"""As regras de um idioma: o JSON que o time de produto edita, já validado.

Regra é dado, não código. Quem muda o comportamento do motor edita um arquivo
JSON, sobe a `version` e publica; o motor não tem nenhum ramo que cite o nome de
uma intenção. Por isso a validação aqui é estrita e cada recusa sai com o caminho
JSON do nó culpado em `details["path"]`: um erro de digitação no arquivo de
regras é ouvido no carregamento, com o endereço do problema, e nunca vira uma
resposta estranha na primeira mensagem de um usuário.

`rule_id` é `f"{intent_id}:{index}"` e é a chave em todo o resto do motor: a
deduplicação do `matching`, cada item do trace, cada resposta da API e cada linha
da tela. Por isso o índice vem da posição no arquivo e não de um campo escrito à
mão, que poderia sair fora de ordem.

O caminho padrão é **dentro do pacote**, `intent/rules/{language}.json`, que é o
que o `package-data` embarca: o mesmo código carrega as mesmas regras de um
checkout e de uma wheel instalada. `resources_dir()` é a casa de
`language_profiles.json` e de mais nada; as duas raízes são deliberadamente
diferentes e nada procura regras em `resources/`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

from intent.errors import RulesError, UnknownIntentError, UnknownLanguageError
from intent.normalize import strip_accents

__all__ = [
    "IntentSpec",
    "Rule",
    "RuleSet",
    "RuleType",
    "load_all_rulesets",
    "load_ruleset",
    "validate_document",
]

_LANGUAGE = re.compile(r"^[a-z]{2}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_INTENT_ID = re.compile(r"^[a-z][a-z0-9_]*$")
# as fronteiras implícitas do tipo `pattern`: sem elas, `assinar` casaria dentro
# de `desassinar` e quem dissesse que desassinou dispararia "assinar"
_LOOKBEHIND = r"(?<!\w)"
_LOOKAHEAD = r"(?!\w)"
# um problema do documento inteiro não tem caminho dentro dele
_DOCUMENT_PATH = ""


class RuleType(StrEnum):
    """Os quatro tipos de regra. São contrato público: o arquivo os nomeia."""

    KEYWORD = "keyword"
    LEMMA = "lemma"
    PATTERN = "pattern"
    ALL = "all"


@dataclass(frozen=True)
class Rule:
    """Uma regra compilada, pronta para o `matching`.

    `rule_id` é `f"{intent_id}:{index}"`, com `index` a posição no arquivo.
    `value` é a string do autor, ou a tupla de `all`; em `pattern` é o valor sem
    acento, o mesmo que `compiled` casa. `compiled` existe só em `pattern`, e
    vale para as demais ser `None`: nenhum outro tipo casa por expressão
    regular, e um campo opcional mente menos do que um regex inerte.
    """

    rule_id: str
    type: RuleType
    value: str | tuple[str, ...]
    weight: float
    compiled: re.Pattern | None


@dataclass(frozen=True)
class IntentSpec:
    """Uma intenção do arquivo, com as regras já numeradas.

    `min_score` é o piso de ponto da intenção (§4.4 do spec): fica ao lado da
    regra que o ganha, porque é política de conteúdo de quem escreve o arquivo.
    `examples` não pontua nada; é o texto de apoio para quem edita o arquivo.
    """

    id: str
    label: str
    priority: int
    min_score: float
    rules: tuple[Rule, ...]
    examples: tuple[str, ...]


@dataclass(frozen=True)
class RuleSet:
    """As regras de um idioma, validadas e congeladas.

    `sha256` é o resumo dos bytes crus do arquivo, então muda em qualquer
    alteração de conteúdo, inclusive de espaço em branco: é a resposta de "o que
    foi carregado mesmo?" para o relatório de recursos. `negators` é tupla
    (aceita frase, como "deixar de") e `negation_boundaries` é conjunto, porque
    a janela de negação só pergunta por pertencimento.
    """

    language: str
    version: str
    intents: tuple[IntentSpec, ...]
    negators: tuple[str, ...]
    negation_boundaries: frozenset[str]
    sha256: str

    def by_id(self, id: str) -> IntentSpec:
        """A intenção `id`, ou `UnknownIntentError` se o arquivo não a descreve.

        Varre a lista em vez de guardar um índice: são onze intenções, e um
        campo `dict` num dataclass congelado é mutável por dentro — o mesmo
        motivo pelo qual `schemas.py` diz que `frozen=True` não é
        imutabilidade profunda.
        """
        for intent in self.intents:
            if intent.id == id:
                return intent
        raise UnknownIntentError(
            f"o arquivo de regras do idioma {self.language!r} não descreve a "
            f"intenção {id!r}",
            {"id": id},
        )


def validate_document(document: dict) -> None:
    """Valida o documento de regras inteiro, na ordem do mais externo ao mais interno.

    A ordem importa: quando dois nós do arquivo estão errados ao mesmo tempo, o
    erro reportado é o do nó mais externo, que é o que vem antes na leitura do
    arquivo. Devolve `None` ou levanta `RulesError`; a mensagem diz o que está
    errado e `details["path"]` diz onde — `"intents.0.rules.1.weight"`, lido por
    quem mantém o arquivo.
    """
    _check(isinstance(document, dict), _DOCUMENT_PATH,
           "o arquivo de regras precisa ser um objeto JSON")
    language = document.get("language")
    _check(isinstance(language, str) and _LANGUAGE.fullmatch(language) is not None,
           "language", "`language` precisa ser um código de duas letras, como em `pt`")
    version = document.get("version")
    _check(isinstance(version, str) and _VERSION.fullmatch(version) is not None,
           "version", "`version` precisa ser semântica, como em `1.0.0`")
    intents = document.get("intents")
    _check(isinstance(intents, list) and len(intents) > 0, "intents",
           "`intents` precisa ser uma lista com pelo menos uma intenção")
    seen: set[str] = set()
    for index, intent in enumerate(intents):
        _check_intent(intent, f"intents.{index}")
        identifier = intent["id"]
        _check(identifier not in seen, f"intents.{index}.id",
               f"a intenção {identifier!r} aparece duas vezes no arquivo de regras")
        seen.add(identifier)
    _check_phrases(document.get("negators"), "negators")
    _check_phrases(document.get("negation_boundaries"), "negation_boundaries")


def load_ruleset(language: str, path: Path | None = None) -> RuleSet:
    """Lê, valida e compila as regras de `language`.

    `path` aponta para o arquivo; omitido, vale o padrão dentro do pacote
    (`intent/rules/{language}.json`). Um arquivo que não está lá é
    `UnknownLanguageError` dizendo qual caminho foi procurado, porque quem
    perguntou precisa saber o que colocar no lugar; um arquivo que está lá e
    está errado é `RulesError` dizendo onde. Os dois são recusados no
    carregamento: o motor é estrito ao carregar e calado em tempo de execução.

    O `language` do `RuleSet` devolvido é o que foi pedido, não o que está
    escrito no arquivo: foi o argumento que escolheu o caminho, e é por ele que
    o relatório de recursos e o dicionário de `load_all_rulesets` são indexados.
    """
    source = _default_path(language) if path is None else path
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise UnknownLanguageError(
            f"o idioma {language!r} não tem um arquivo de regras legível "
            f"({exc.strerror}); o caminho esperado é {source}"
        ) from exc
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RulesError(
            f"o arquivo de regras em {source} não é JSON válido: {exc}",
            {"path": _DOCUMENT_PATH},
        ) from exc
    validate_document(document)
    return _build(language, document, hashlib.sha256(raw).hexdigest())


def load_all_rulesets(languages: tuple[str, ...]) -> dict[str, RuleSet]:
    """As regras de cada idioma de `languages`, na ordem em que foram pedidos.

    A primeira falta interrompe a carga: um motor com metade dos idiomas
    carregados é um motor que responde errado em vez de recusar, e o
    `UnknownLanguageError` diz qual arquivo não estava lá.
    """
    return {language: load_ruleset(language) for language in languages}


def _default_path(language: str) -> Path:
    """O caminho das regras de `language` dentro do pacote.

    Junto do módulo, e não em `resources/`, porque é este caminho que o
    `package-data` embarca na wheel: em `resources/` o arquivo só existiria num
    checkout, e a instalação passaria a responder que não tem o idioma.
    """
    return Path(__file__).resolve().parent / "rules" / f"{language}.json"


def _check_intent(intent, path: str) -> None:
    """Valida uma entrada de `intents`, do `id` às regras."""
    _check(isinstance(intent, dict), path, "cada entrada de `intents` precisa ser um objeto")
    identifier = intent.get("id")
    # `fullmatch` e não `match`: o `$` de `^[a-z][a-z0-9_]*$` também casa antes
    # de um `\n` no fim, e um id com quebra de linha escaparia da regra
    _check(isinstance(identifier, str)
           and _INTENT_ID.fullmatch(identifier) is not None,
           f"{path}.id", "`id` precisa ser minúsculo, começando por letra, como em `cancelar`")
    label = intent.get("label")
    _check(isinstance(label, str) and label.strip() != "", f"{path}.label",
           "`label` precisa de um texto não vazio, é o que a interface mostra")
    priority = intent.get("priority")
    _check(isinstance(priority, int) and not isinstance(priority, bool),
           f"{path}.priority", "`priority` precisa ser um número inteiro")
    _check_positive(intent.get("min_score"), f"{path}.min_score", "`min_score`")
    rules = intent.get("rules")
    _check(isinstance(rules, list) and len(rules) > 0, f"{path}.rules",
           "`rules` precisa ser uma lista com pelo menos uma regra")
    _check_rules(rules, f"{path}.rules")
    examples = intent.get("examples")
    _check(isinstance(examples, list)
           and all(isinstance(item, str) and item.strip() != "" for item in examples),
           f"{path}.examples", "`examples` precisa ser uma lista de textos")


def _check_rules(rules: list, path: str) -> None:
    """Valida as regras de uma intenção e recusa `type`+`value` repetidos.

    A chave da repetição é o par, não o valor: a mesma palavra como `keyword` e
    como `lemma` casa com coisas diferentes e continua válida. O que não pode
    repetir é o par, porque as duas regras contariam o mesmo token duas vezes.
    """
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for index, rule in enumerate(rules):
        rule_path = f"{path}.{index}"
        _check(isinstance(rule, dict), rule_path, "cada regra precisa ser um objeto")
        kind = _rule_type(rule.get("type"), f"{rule_path}.type")
        raw = rule.get("value")
        value = _check_rule_value(kind, raw, f"{rule_path}.value")
        _check_positive(rule.get("weight"), f"{rule_path}.weight", "`weight`")
        key = (kind.value, value)
        _check(key not in seen, rule_path,
               f"a regra {kind.value} {raw!r} já aparece nesta intenção, "
               "e as duas contariam o mesmo token em dobro")
        seen.add(key)


def _rule_type(raw, path: str) -> RuleType:
    """O tipo da regra, ou recusa listando os quatro que existem."""
    _check(isinstance(raw, str), path, "`type` precisa ser o nome de um tipo de regra")
    try:
        return RuleType(raw)
    except ValueError:
        _reject(path, f"tipo de regra desconhecido: {raw!r}; use um de "
                      f"{', '.join(kind.value for kind in RuleType)}")


def _check_rule_value(kind: RuleType, raw, path: str) -> tuple[str, ...]:
    """Confere o `value` de uma regra e o devolve como tupla de valores.

    `keyword` e `lemma` são uma palavra só: um valor com espaço é uma frase, e
    frase é o que `pattern` existe para casar. `all` exige duas ou mais, porque
    existe para exigir dois sinais independentes, e também uma palavra só por
    valor, para o consumo de token ficar bem definido.

    Todo valor passa por uma compilação de expressão regular, mesmo sem ser
    `pattern`: um valor que não compila é quase sempre um erro de digitação
    (`[nao fechada`), e quem escreve o arquivo precisa ouvir isso no
    carregamento. Só `pattern` é conferido também dentro das fronteiras de
    palavra, porque é o único que compila de fato.
    """
    if kind is RuleType.ALL:
        _check(isinstance(raw, list) and len(raw) >= 2, path,
               "`all` precisa de uma lista com pelo menos duas palavras")
        for index, item in enumerate(raw):
            _check_single_word(item, f"{path}.{index}")
            _check_regex(item, f"{path}.{index}")
        return tuple(raw)
    _check(isinstance(raw, str), path, f"`{kind.value}` precisa de um texto")
    if kind is RuleType.PATTERN:
        # o valor é conferido já sem acento, que é a forma que será compilada
        pattern = strip_accents(raw)
        _check_regex(pattern, path)
        _check_regex(_wrapped(pattern), path)
        return (pattern,)
    _check_single_word(raw, path)
    _check_regex(raw, path)
    return (raw,)


def _check_single_word(raw, path: str) -> None:
    """Recusa um valor que não é uma palavra só, sem espaço."""
    _check(isinstance(raw, str), path, "o valor precisa ser um texto")
    _check(raw.strip() != "" and not any(char.isspace() for char in raw), path,
           "o valor precisa ser uma palavra só, sem espaço: uma frase é do tipo `pattern`")


def _check_regex(value: str, path: str) -> None:
    """Recusa `value` se ele não compilar como expressão regular."""
    try:
        re.compile(value)
    except re.error as exc:
        _reject(path, f"o valor não compila como expressão regular: {exc}")


def _check_positive(raw, path: str, label: str) -> None:
    """Confere um número que precisa ser finito e maior que zero.

    `float()` dentro de um `try`, porque quem edita o arquivo escreve `2` e não
    `2.0`, e recusar isso seria uma forma cara de gastar a tarde de quem não
    escreve Python; por isso `"2"` também entra, é a mesma mão no mesmo teclado.
    O que não passa é o que não serve: `nan` e `inf` não são ponto, e
    `nan <= 0` é falso — uma conferência só de positividade deixaria o `nan`
    passar e ele envenenaria toda a pontuação depois. `bool` é recusado antes,
    pelo mesmo motivo que o `priority`: `float(True)` é `1.0`, e um `true` no
    arquivo é erro de digitação, não um peso de um ponto.
    """
    if isinstance(raw, bool):
        _reject(path, f"{label} precisa ser um número maior que zero, recebido {raw!r}")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        _reject(path, f"{label} precisa ser um número maior que zero, recebido {raw!r}")
    if not math.isfinite(number) or number <= 0:
        _reject(path, f"{label} precisa ser um número maior que zero, recebido {raw!r}")


def _check_phrases(raw, path: str) -> None:
    """Confere `negators` e `negation_boundaries`.

    Frase é permitida aqui — "deixar de" é um negador inteiro — mas a lista de
    textos é o formato, porque é assim que o arquivo é escrito, e uma chave
    errada (`"negators": {"nao": true}`) não pode passar em silêncio.
    """
    _check(isinstance(raw, list)
           and all(isinstance(item, str) and item.strip() != "" for item in raw),
           path, "precisa ser uma lista de textos não vazios")


def _wrapped(value: str) -> str:
    """`value` entre as fronteiras de palavra implícitas do tipo `pattern`."""
    return f"{_LOOKBEHIND}{value}{_LOOKAHEAD}"


def _build(language: str, document: dict, sha256: str) -> RuleSet:
    """Monta o `RuleSet` de um documento já validado."""
    return RuleSet(
        language=language,
        version=document["version"],
        intents=tuple(_intent_spec(intent) for intent in document["intents"]),
        negators=tuple(document["negators"]),
        negation_boundaries=frozenset(document["negation_boundaries"]),
        sha256=sha256,
    )


def _intent_spec(intent: dict) -> IntentSpec:
    """Monta a `IntentSpec` de uma entrada de `intents` já validada.

    O `rule_id` sai da posição da regra na lista, e não de um campo do arquivo:
    ele é a chave de rastreio de todo o motor, e um campo escrito à mão
    poderia vir trocado, repetido ou fora de ordem sem nada reclamar.
    """
    identifier = intent["id"]
    return IntentSpec(
        id=identifier,
        label=intent["label"],
        priority=intent["priority"],
        min_score=float(intent["min_score"]),
        rules=tuple(
            _rule(f"{identifier}:{index}", rule)
            for index, rule in enumerate(intent["rules"])
        ),
        examples=tuple(intent["examples"]),
    )


def _rule(rule_id: str, rule: dict) -> Rule:
    """Monta a `Rule` de um objeto de regra já validado.

    Só `pattern` ganha regex compilada, e o valor perde o acento antes de
    virar regex: a mensagem chega ao `matching` normalizada, em minúsculas e sem
    acento, então um padrão escrito `não\\s+funciona` ainda casa com
    `nao funciona` e quem escreve o arquivo não precisa pensar em acento. O
    `value` guardado é o mesmo já sem acento que o regex casa, para o trace
    mostrar o que rodou de verdade.
    """
    kind = RuleType(rule["type"])
    weight = float(rule["weight"])
    raw = rule["value"]
    if kind is RuleType.ALL:
        return Rule(rule_id, kind, tuple(raw), weight, None)
    if kind is RuleType.PATTERN:
        pattern = strip_accents(raw)
        # a mesma expressão que `_check_rule_value` já compilou, então aqui não
        # pode falhar
        return Rule(rule_id, kind, pattern, weight, re.compile(_wrapped(pattern)))
    return Rule(rule_id, kind, raw, weight, None)


def _check(ok: bool, path: str, message: str) -> None:
    """Falha em `path` quando `ok` é falso, para toda recusa ter o mesmo formato."""
    if not ok:
        raise RulesError(message, {"path": path})


def _reject(path: str, message: str) -> NoReturn:
    """Recusa em `path`, para os casos em que a mensagem é montada no próprio ponto."""
    raise RulesError(message, {"path": path})

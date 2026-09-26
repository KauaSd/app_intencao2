"""Identificação de idioma em duas etapas: roteiro, depois palavras funcionais.

A mensagem é normalizada antes de contar qualquer coisa, para que o contador
veja exatamente a forma que os padrões de regra veem: minúsculas, sem
acento, sem pontuação. Nenhum idioma é devolvido como `supported` sem passar
pelos dois portões, e cada recusa sai com um motivo explícito
(`empty_text`, `insufficient_signal`, `ambiguous_language`,
`unsupported_script`) em vez de um palpite. "취소해줘" é coreano e não
português: sem o portão de roteiro ele acabaria numa resposta que nenhum
arquivo de regras consegue sustentar (spec §6.1, §6.2).

Os marcadores de `resources/language_profiles.json` são palavras inteiras,
minúsculas e sem acento, que é a forma que `normalise` produz. Só os
`exclusive` contam: `shared` documenta o vocabulário que os dois idiomas
usam e por isso nunca é evidência de nenhum deles.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from intent.config import Settings, resources_dir
from intent.normalize import has_cased_character, normalise
from intent.schemas import LanguageCandidate, LanguageInfo

__all__ = ["LanguageIdentifier", "Reason", "load_profiles"]

_PROFILES_FILE = "language_profiles.json"

# primeira palavra de `unicodedata.name`, que é o que separa a família de um
# caractere do resto do nome ("LATIN SMALL LETTER A" -> LATIN)
_SCRIPT_BY_NAME = {
    "LATIN": "Latin",
    "CJK": "CJK",
    "HIRAGANA": "CJK",
    "KATAKANA": "CJK",
    "HANGUL": "CJK",
    "CYRILLIC": "Cyrillic",
    "ARABIC": "Arabic",
}
_UNKNOWN_SCRIPT = "Unknown"


class Reason:
    """Motivo do veredito. São contrato público: a API os expõe."""

    OK: str = "ok"
    INSUFFICIENT_SIGNAL: str = "insufficient_signal"
    AMBIGUOUS_LANGUAGE: str = "ambiguous_language"
    UNSUPPORTED_SCRIPT: str = "unsupported_script"
    EMPTY_TEXT: str = "empty_text"


def load_profiles(path: Path | None = None) -> dict:
    """Lê o documento de perfis de `path`, ou de `resources/` quando omitido.

    Devolve o documento como está no disco. Quem chama decide o que fazer com
    um idioma que o documento não descreve: o identificador o trata como sem
    marcadores, e um perfil só com `shared` não prova nada.
    """
    source = resources_dir() / _PROFILES_FILE if path is None else path
    return json.loads(source.read_text(encoding="utf-8"))


class LanguageIdentifier:
    """Conta palavras funcionais por idioma, depois de conferir o roteiro.

    `profiles` é o documento de `load_profiles` e `settings` diz quais idiomas
    o motor atende. Os marcadores são lidos uma vez, na construção, e o
    identificador não guarda estado entre chamadas: dois usos do mesmo objeto
    não podem discordar um do outro.
    """

    def __init__(self, profiles: dict, settings: Settings) -> None:
        self._settings = settings
        described = profiles.get("languages", {})
        self._markers = {
            language: frozenset(described.get(language, {}).get("exclusive", ()))
            for language in settings.languages
        }
        # um token é decisivo se algum idioma o reconhece; o mesmo token
        # reconhece os dois quando a palavra existe nos dois idiomas
        self._evidence = frozenset().union(*self._markers.values())

    def identify(self, text: str) -> LanguageInfo:
        """O veredito de idioma para `text`, decidido sem adivinhar.

        A ordem é a do spec §6: nada de texto reconhecível, depois roteiro,
        depois perfis. Cada recusa devolve os candidatos que a contagem
        produziu, porque uma recusa que não se explica não serve para nada.

        As duas medidas que acompanham a recusa têm sentidos diferentes.
        `min_share` é a política de `Settings` — o piso que a mensagem não
        alcança — e é o mesmo número nas duas recusas de contagem, para que a
        interface não precise saber qual delas ocorreu. `margin` é o que a
        mensagem mediu: a diferença entre o primeiro e o segundo colocado, e
        `0.0` quando ninguém disputa a liderança.
        """
        n = normalise(text)
        if not has_cased_character(n.text):
            # longo o bastante para passar pelo portão de comprimento e ainda
            # assim sem nada que se possa contar; sem texto não há
            # deslocamento para reportar
            return LanguageInfo(None, False, Reason.EMPTY_TEXT, None, True, ())
        script, latin_share = _script_shares(n.text)
        offset_preserved = n.index_map == tuple(range(len(n.text)))
        if latin_share < self._settings.min_script_share:
            # nenhum arquivo de regras existe para outra família de roteiro,
            # então adivinhar a partir de três palavras emprestadas só produziria
            # uma resposta errada e confiante
            return LanguageInfo(None, False, Reason.UNSUPPORTED_SCRIPT, script,
                                offset_preserved, ())
        candidates = self._candidates(_word_tokens(n.text))
        top = candidates[0].share if candidates else 0.0
        second = candidates[1].share if len(candidates) > 1 else 0.0
        margin = top - second
        if not candidates or top < self._settings.min_language_share:
            return LanguageInfo(None, False, Reason.INSUFFICIENT_SIGNAL, script,
                                offset_preserved, candidates,
                                self._settings.min_language_share, margin)
        if margin < self._settings.min_language_margin:
            return LanguageInfo(None, False, Reason.AMBIGUOUS_LANGUAGE, script,
                                offset_preserved, candidates,
                                self._settings.min_language_share, margin)
        return LanguageInfo(candidates[0].language, True, Reason.OK, script,
                            offset_preserved, candidates)

    def _candidates(self, tokens: tuple[str, ...]) -> tuple[LanguageCandidate, ...]:
        """A participação de cada idioma configurado nos tokens decisivos.

        A participação é `marcadores / tokens decisivos`, e um token decisivo
        é um que pelo menos um idioma reconhece. Um token que os dois
        reconhecem conta para os dois e uma vez só no denominador, que é o
        que permite a soma passar de 1 numa frase genuinamente misturada. Um
        idioma abaixo de `mixed_language_ratio` sai da lista: uma palavra em
        inglês numa frase portuguesa é vocabulário emprestado, não mistura.
        """
        decision = sum(1 for token in tokens if token in self._evidence)
        if not decision:
            return ()
        candidates: list[LanguageCandidate] = []
        for language, markers in self._markers.items():
            found = tuple(token for token in tokens if token in markers)
            share = len(found) / decision
            if share < self._settings.mixed_language_ratio:
                continue
            candidates.append(LanguageCandidate(language, share, found))
        return tuple(sorted(candidates, key=lambda c: (-c.share, c.language)))


def _script_shares(text: str) -> tuple[str, float]:
    """A família de roteiro dominante e a participação do latim.

    Só caractere com letra entra na conta: dígito, pontuação e emoji não têm
    família e não devem empurrar o latim para baixo. Chamada depois do portão
    `has_cased_character`, então a soma nunca é zero. Um caractere cujo nome
    não é de família nenhuma conta como `Unknown`, que derruba a participação
    do latim em vez de passar como se fosse latim.
    """
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "").split(" ", 1)[0]
        family = _SCRIPT_BY_NAME.get(name, _UNKNOWN_SCRIPT)
        counts[family] = counts.get(family, 0) + 1
    dominant = sorted(counts, key=lambda family: (-counts[family], family))[0]
    return dominant, counts.get("Latin", 0) / sum(counts.values())


def _word_tokens(text: str) -> tuple[str, ...]:
    """O texto normalizado em palavras, sem número e sem pontuação.

    Separador caractere a caractere, como em `normalise`: "erro500" é a
    palavra "erro" seguida de um número, e não uma palavra nova que nenhum
    perfil conhece.
    """
    spaced = "".join(char if char.isalpha() else " " for char in text)
    return tuple(spaced.split())

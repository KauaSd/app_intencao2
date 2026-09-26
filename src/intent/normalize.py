"""A cópia normalizada da mensagem e o mapa que a liga ao texto original.

Os padrões casam contra a cópia normalizada, mas `char_start`/`char_end` são
reportados no texto que o usuário digitou. `normalise` devolve as duas coisas
juntas, para que nenhum deslocamento dependa de uma suposição sobre o tamanho
da transformação.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

__all__ = [
    "Normalised",
    "has_cased_character",
    "normalise",
    "strip_accents",
    "to_original_offset",
]


@dataclass(frozen=True)
class Normalised:
    """Texto normalizado e o mapa de volta para o texto original.

    `index_map[i]` é o índice, no texto original, do caractere que produziu
    `text[i]`. `len(index_map) == len(text)` sempre: os dois são construídos
    na mesma passagem, caractere a caractere, e aparados juntos.
    """

    text: str
    index_map: tuple[int, ...]


def normalise(text: str) -> Normalised:
    """Normaliza `text` e devolve o texto com o mapa de índices.

    Minúsculas, NFD sem marcas combinantes, uma sequência de espaço virando
    um espaço, e nada de espaço nas pontas. A pontuação da frase fica, para
    que os padrões possam confiar em fronteiras `.?!`.
    """
    chars: list[str] = []
    index_map: list[int] = []
    in_space_run = False
    for index, char in enumerate(text):
        if char.isspace():
            if not in_space_run:
                # a sequência inteira vira um só espaço, com o índice do primeiro
                chars.append(" ")
                index_map.append(index)
            in_space_run = True
            continue
        stripped = _strip_marks(char.lower())
        if not stripped:
            # um caractere que desaparece, como uma marca combinante solta,
            # não separa duas sequências de espaço
            continue
        in_space_run = False
        for emitted in stripped:
            # uma decomposição pode emitir mais de um caractere, e todos saem
            # do mesmo índice do original
            chars.append(emitted)
            index_map.append(index)
    return _trimmed(chars, index_map)


def to_original_offset(n: Normalised, offset: int) -> int:
    """Traduz um índice de `n.text` para um índice do texto original.

    Índice fora do mapa é fixado na ponta mais próxima em vez de levantar
    exceção, porque um padrão pode terminar legitimamente em `len(n.text)`. A
    ponta é a posição logo depois do último caractere mapeado, e um mapa vazio
    devolve 0. O que esse limite significa é decisão do pipeline, que marca
    `offsets_exact = False` nesses casos.
    """
    if offset < 0:
        return 0
    if offset < len(n.index_map):
        return n.index_map[offset]
    return n.index_map[-1] + 1 if n.index_map else 0


def strip_accents(text: str) -> str:
    """NFD sem marcas combinantes, sem mexer nas maiúsculas.

    Usada no carregamento de regras: o valor do padrão perde o acento e
    preserva a caixa, porque quem decide a caixa é `normalise`, sobre o texto
    da mensagem. Caractere a caractere, como lá: a mesma entrada não pode
    virar uma string num caminho e outra no outro, e é isso que faz um padrão
    de regra casar com a mensagem.
    """
    return "".join(_strip_marks(char) for char in text)


def has_cased_character(text: str) -> bool:
    """True quando algum caractere é alfabético e não é marca combinante.

    Um texto só com dígitos, pontuação ou espaços passa no portão de
    comprimento e ainda assim não diz nada: é `empty_text`, não um palpite.
    """
    return any(char.isalpha() and not unicodedata.combining(char) for char in text)


def _strip_marks(text: str) -> str:
    """NFD, descarta as marcas combinantes e recompõe em NFC.

    O NFC final não é enfeite: hangul decompõe em jamos que não são marcas
    combinantes, então sem ele `취소해줘` voltaria decomposto. Recompor só o
    que já cabia em um caractere é o que impede que um jamo e o seguinte se
    juntem em um único caractere, e portanto o que mantém
    `len(index_map) == len(text)`.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


def _trimmed(chars: list[str], index_map: list[int]) -> Normalised:
    """Junta as listas em um `Normalised`, aparando os dois lados juntos."""
    start, end = 0, len(chars)
    while start < end and chars[start] == " ":
        start += 1
    while end > start and chars[end - 1] == " ":
        end -= 1
    return Normalised("".join(chars[start:end]), tuple(index_map[start:end]))

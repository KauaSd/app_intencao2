"""A avaliação das regras de um idioma: quatro tipos, consumo de token e uma
pontuação por regra — e, no fim do mesmo arquivo, a etapa que ordena o que a
pontuação produziu.

Este módulo é o núcleo de pontuação do motor. Ele recebe o `Doc` do spaCy e o
`Normalised` da mesma mensagem e devolve as `IntentScore` que casaram — sem
filtrar por `min_score`, porque filtrar é trabalho da etapa seguinte e é
justamente por não filtrar aqui que um teste consegue olhar a pontuação crua.
Uma intenção que não casou não aparece: o resultado é a lista do que o motor
viu, não um palpite sobre o que ele deveria ter visto. `rank`, no fim do
arquivo, é essa etapa seguinte, e ela mora aqui porque o `File Map` do plano
divide o trabalho em `matching.py` (pontuar) e `pipeline.py` (montar o
resultado): quem lê `evaluate` e quem lê `rank` precisam do mesmo `RuleSet` e
do mesmo `Settings` na mão, e um módulo só para ordenar duas listas já
carregadas custaria um arquivo e um import para dizer a mesma coisa.

Três decisões explicam quase todo o resto.

**A avaliação são duas passagens, e a ordem entre elas é o contrato.** O tipo
`all` exige dois sinais independentes e, ao exigir, *consome* os tokens que
encontrou: sem consumo, `all ["cancelar","assinatura"]` junto com
`keyword "cancelar"` pontuaria 5.5 por uma menção só da palavra. Consumir
durante uma passagem única, na ordem em que as regras estão no arquivo, daria
3.5 com o `all` antes e 5.5 com o `keyword` antes — e a pontuação passaria a
depender de como o JSON foi digitado. Arquivo de regra é conteúdo, e a ordem
das linhas de um conteúdo não muda o resultado. Por isso as regras `all` rodam
inteiras primeiro, e todo o resto depois, pulando qualquer token já consumido —
um `pattern` que passa por cima de um token consumido é descartado inteiro, e
não só o token repetido, porque o que sobraria dele seria um token qualquer do
trecho e não a evidência que ele representa.
Duas regras `all` que se sobrepõem não se excluem: excluí-las aí devolveria a
mesma dependência de ordem para dentro da primeira passagem, e o consumo existe
para impedir que uma regra de token único reconte o que a conjunção já
afirmou, não para arbitrar entre duas conjunções.

**Uma regra pontua uma vez, e o trace mostra o que dá para mostrar.** A
pontuação é a soma do `weight` das regras que casaram, uma vez cada:
`cancelar cancelar cancelar` pontua o mesmo que `cancelar`, e a repetição não
vira ponto. O trace, esse, não é a foto completa do texto: `keyword`, `lemma` e
`pattern` deixam um acerto por ocorrência, mas um `all` deixa um acerto por
palavra casada e nada mais — a segunda menção da mesma palavra foi consumida na
primeira passagem e por isso não chega a ser contada por ninguém. Por isso a
soma é por `rule_id` e a deduplicação é por par `(rule_id, token)`: um `RawHit`
repetido do mesmo `rule_id` com o mesmo token é a mesma evidência dita duas
vezes.

**A negação vem depois das duas passagens, e por ocorrência.** A janela de
negação (§5.2) é uma terceira passagem, sobre os acertos que as duas primeiras
juntaram: varrer para a esquerda no meio da avaliação puniria a ordem das linhas
do arquivo de novo, e um `all` só tem as duas palavras depois de as duas terem
sido encontradas. A varredura parte do *primeiro token do acerto* e olha para
tras, o que torna a negação uma propriedade da ocorrência e não da regra: a
mesma palavra dita duas vezes, uma negada e outra não, é metade evidência. Isso
vale para `keyword`, `lemma` e `pattern`, e não vale para `all`.

Por isso a soma é refeita ao final em vez de subtraída, e o peso que sobra de
uma regra depende do tipo dela. `keyword`, `lemma` e `pattern` só perdem o peso
quando *todos* os seus acertos foram negados: são afirmações independentes, e
negar uma não diz nada sobre as outras. Um `all` perde o peso assim que *um* dos
seus acertos é negado, porque os acertos dele não são afirmações independentes e
sim os componentes de uma conjunção só — `all ["cancelar","assinatura"]` em
"não cancelar, mas assinatura" não afirma a conjunção, e deixar o peso inteiro
passaria do `min_score` da própria intenção. Nos dois casos `RawHit.negated`
continua visível no trace, que é o que separa "o motor errou" de "o motor decidiu
certo e mostrou por quê".

**A ordem é total, e o que fecha a conta é o id.** `rank` ordena por
`(-score, -priority, intent_id)`: o ponto manda, a prioridade desempata, e o id
desempata o desempate. O id é a última chave porque duas intenções com o mesmo
ponto e a mesma prioridade precisam parar em algum lugar fixo — sem ele a ordem
seria a do arquivo de regras ou a da soma, e a tabela de
esperado-versus-detectado da interface piscaria entre duas execuções da mesma
mensagem. O `min_score` que essa etapa usa é o da própria `IntentScore`, que o
carregou do `RuleSet`: é política de conteúdo de quem escreve o arquivo de
regras, e nenhum `Settings` entra nessa conta. O que o `Settings` traz é o teto
de `max_intents` e o piso de `min_confidence`, e este último é aplicado **só ao
vencedor** — reprovar o primeiro colocado por confiança e promover o segundo
trocaria "o motor não achou nada de confiável" por "o motor achou e confia no
segundo", que é a resposta errada para a pergunta errada. Quando o vencedor não
passa, o resultado é `desconhecido` com `kept` vazio, e o quase-acerto continua
inteiro em `dropped`.

**Ponto zero não é candidato.** A negação zera a intenção sem apagá-la: é o que
permite à interface dizer "o motor viu `cancelar` e o nebou". Um `min_score` de
zero deixaria essa evidência negada virar resposta, e `rules.py` recusa
`min_score <= 0` no carregamento justamente por isso — o `score > 0.0` aqui é a
mesma regra defendida de novo, para o `IntentScore` montado à mão que um teste
constrói e para o piso que ninguém deveria escrever. A intenção zerada vai para
`dropped`, nunca para `kept`.

O resto do módulo cuida do resto do texto. O `Doc` foi construído sobre a
mensagem que o usuário digitou, então `token.idx` e `char_start`/`char_end` são
índices do mesmo texto e a comparação entre eles é direta; o `pattern`, esse,
casa contra a cópia normalizada e atravessa `to_original_offset` para voltar ao
texto original. Os atributos com underscore do spaCy (`lower_`, `lemma_`,
`idx`) não saem daqui: o que este módulo publica são `RawHit`, `IntentScore`,
`RankedResult`, `IntentMatch` e `RuleView`, e nenhum deles carrega um token do
spaCy — só os índices e o texto, que já são o texto que o usuário digitou. Quem
precisar de um token anotado recebe `TokenView` de outro lugar.
"""

from __future__ import annotations
from bisect import bisect_left
from collections.abc import Callable, Iterable

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.config import Settings
from intent.normalize import Normalised, strip_accents, to_original_offset
from intent.rules import IntentSpec, Rule, RuleSet, RuleType
from intent.schemas import IntentMatch, RuleView

if TYPE_CHECKING:
    from spacy.tokens import Doc, Token

__all__ = [
    "UNKNOWN_INTENT",
    "IntentScore",
    "Matcher",
    "RankedResult",
    "RawHit",
    "confidence_for",
    "rank",
]

# o rótulo que o motor emite quando não tem intenção a relatar. Ele é de
# propósito ausente do catálogo (§7 do spec): uma intenção que precisasse
# disputar ponto com as outras nunca poderia significar com segurança "nenhuma
# intenção", e é por isso que ele é emitido aqui e não escrito em arquivo
UNKNOWN_INTENT = "desconhecido"
# os dois motivos que `rank` sabe relatar. `"ok"` com `kept` vazio é "não
# passou do `min_score` de ninguém"; `"low_confidence"` é "passou e o piso do
# sistema não confiou", e são afirmações diferentes sobre a mesma resposta
_OK = "ok"
_LOW_CONFIDENCE = "low_confidence"

# o que fecha uma oração, e não vem do arquivo de regras: ver `_is_clause_break`
_CLAUSE_BREAKS = frozenset({".", ",", ";", ":", "!", "?"})


@dataclass
class RawHit:
    """Uma ocorrência de regra, com os deslocamentos no texto que o usuário
    digitou.

    É mutável de propósito: a passagem de negação marca `negated` nela depois
    que a pontuação já foi somada, e um dataclass congelado transformaria
    "marcar o que foi negado" em "reconstruir o trace inteiro".

    `char_start`/`char_end` são índices do texto original, e é por eles que a
    interface destaca o trecho. `token_indices` são as posições dos tokens que
    o acerto cobre: é delas que a tabela de tokens tira o `matched` e que a
    negação tira o ponto de partida da varredura para a esquerda. `value` é o
    valor da regra que casou — no tipo `all`, a palavra desta ocorrência e não a
    lista, porque a lista é da regra e a ocorrência é desta palavra.
    """

    rule_id: str
    type: RuleType
    value: str
    weight: float
    char_start: int
    char_end: int
    matched_text: str
    offsets_exact: bool
    token_indices: tuple[int, ...]
    negated: bool = False


@dataclass(frozen=True)
class IntentScore:
    """A pontuação de uma intenção, com o trace que a produziu.

    `score` é a soma dos pesos das regras que casaram, e `hits` é o trace
    completo, inclusive das ocorrências que a negação vai zerar: é a diferença
    entre "o motor errou" e "o motor decidiu certo e mostrou por quê".
    """

    intent_id: str
    label: str
    score: float
    priority: int
    min_score: float
    hits: tuple[RawHit, ...]


class Matcher:
    """As regras de um idioma viradas em pontuação, para uma mensagem.

    Não guarda estado entre chamadas: o mesmo `Matcher` atende a fila do
    pipeline, e um `evaluate` que lembrasse da mensagem anterior transformaria
    uma dependência que o chamador injeta numa dependência implícita. Também
    não lê disco nem ambiente — o `RuleSet` e o `Settings` chegam prontos na
    construção.
    """

    def __init__(self, ruleset: RuleSet, settings: Settings) -> None:
        """Guarda as regras e a configuração da detecção.

        A janela de negação é montada aqui, e não a cada `evaluate`: o `RuleSet` e
        o `Settings` chegam imutáveis e a tabela de negadores sai deles inteira,
        então não existe por que refazê-la mensagem a mensagem. O `settings` é lido
        por `negation_window` e é essa leitura que transforma a dependência
        explícita numa configuração que muda de verdade.
        """
        self._ruleset = ruleset
        self._settings = settings
        self._negation = _build_negation(
            ruleset.negators, ruleset.negation_boundaries,
            self._settings.negation_window)

    def evaluate(self, doc: Doc, normalised: Normalised) -> list[IntentScore]:
        """Pontua cada intenção de `doc` contra `normalised`, na ordem do
        arquivo.

        Devolve uma entrada por intenção que casou, inclusive as que ficaram
        abaixo do próprio `min_score`: o piso é da etapa que filtra, e é
        olhando a pontuação crua que os testes deste módulo conseguem afirmar
        que o consumo, a deduplicação e a negação funcionaram.
        """
        lexical = _Lexical.build(
            tuple(_negation_form(token.text) for token in doc))
        scores: list[IntentScore] = []
        for intent in self._ruleset.intents:
            evaluation = _IntentEvaluation(
                doc, normalised, intent, self._negation, lexical)
            evaluation.run()
            if not evaluation.hits:
                continue
            scores.append(IntentScore(
                intent_id=intent.id,
                label=intent.label,
                score=evaluation.score,
                priority=intent.priority,
                min_score=intent.min_score,
                hits=tuple(evaluation.hits),
            ))
        return scores


@dataclass(frozen=True)
class RankedResult:
    """A resposta de `rank`, e tudo o que ela deixou de fora.

    `kept` é a resposta: no máximo `Settings.max_intents` intenções, na ordem
    total, e a primeira delas é o `primary`. `dropped` é o diagnóstico — as
    intenções que o motor viu e não relatou, com a pontuação e o trace delas
    intactos, e é ele que a interface e a mensagem de falha do teste de
    fixtures usam para dizer *por que* a resposta ficou mais pobre do que a
    evidência permitiria.

    `confidence` é a do `primary`, e `primary == UNKNOWN_INTENT` sempre vem com
    `confidence == 0.0`: a confiança de uma intenção que o motor decidiu não
    relatar não é uma confiança, e reportar a do quase-acerto faria a barra da
    interface afirmar o contrário do que a resposta diz. `reason` diz por que a
    lista ficou vazia, e é ele que separa "o motor não achou nada" de "o motor
    achou e não confiou": nos dois casos `primary` é `desconhecido` e `kept` é
    vazio, e as duas coisas que faltam para responder são diferentes.
    """

    kept: tuple[IntentMatch, ...]
    dropped: tuple[IntentScore, ...]
    primary: str
    multi_intent: bool
    confidence: float
    reason: str


def confidence_for(score: float) -> float:
    """A confiança de um `score`: `score / (score + 1)`, arredondada a 4 casas.

    A curva satura em 1.0 sem nunca chegar lá, que é o que serve a um número
    que a interface mostra como barra: um `score` de 10.0 e um de 50.0 são os
    dois "praticamente certo" e precisam parecer iguais para quem lê.

    O arredondamento é de exibição e mora aqui, e não em `IntentScore.score`,
    porque `score` é comparado com dois pisos — o `min_score` da intenção e o
    `min_confidence` do sistema — e arredondar antes de comparar trocaria cada
    um desses cortes por um corte diferente, a cada casa.
    """
    return round(score / (score + 1.0), 4)


def rank(scores: list[IntentScore], settings: Settings) -> RankedResult:
    """Filtra, ordena, corta no teto e decide se o que sobrou é confiável.

    A ordem das quatro etapas é o contrato, e cada uma usa o dado que a
    anterior produziu. O `min_score` de cada intenção é o dela, lido da
    `IntentScore` que o carregou do `RuleSet`, e é o filtro que faz o trabalho
    de verdade: o que sobra dele é a lista de intenções que a mensagem
    realmente afirma. A ordenação é total, `(-score, -priority, intent_id)`, e
    vem antes do teto porque o teto corta a cauda e a cauda só existe depois de
    ordenar. O teto é `settings.max_intents`, com um piso de 1 — um teto zero
    viraria um erro de configuração num `IndexError` no caminho quente de cada
    mensagem, e a única leitura possível dele ("não relate nada nunca") é a
    mesma que `min_confidence = 1.0` já expressa, honestamente.

    O portão de confiança vem por último e vale só para o vencedor, porque é o
    vencedor que a interface mostra e é sobre ele que o chatbot vai agir. Se ele
    não passa, o resultado é `desconhecido` com `kept` vazio e **sem** promover
    o segundo colocado: promover trocaria "não achei nada confiável" por
    "achei, e é o outro", que é a resposta errada para a pergunta errada. O
    ponto recusado continua em `dropped`, com o trace inteiro, e é dali que a
    interface mostra o quase-acerto.

    `dropped` é a lista completa do que o motor não relata, e por isso inclui
    também o que o teto cortou: uma intenção que passou do próprio `min_score` e
    ficou de fora só porque `max_intents` é 3 é exatamente o quase-acerto que a
    tela de diagnóstico precisa mostrar, e `RankedResult` não tem outro lugar
    onde ele caberia. A ordem é a mesma chave total, então a lista de
    diagnóstico não volta a depender da ordem em que as intenções casaram.
    """
    ordered = sorted(scores, key=_ranking_key)
    candidates = [score for score in ordered if _clears_floor(score)]
    rejected = [score for score in ordered if not _clears_floor(score)]
    if not candidates:
        return RankedResult(kept=(), dropped=tuple(rejected), primary=UNKNOWN_INTENT,
                            multi_intent=False, confidence=0.0, reason=_OK)
    # `max(1, ...)` pelo motivo do docstring: o teto limita o tamanho da
    # resposta, e um teto de zero não descreve nenhuma política de conteúdo
    kept_scores = candidates[:max(1, settings.max_intents)]
    confidence = confidence_for(kept_scores[0].score)
    if confidence < settings.min_confidence:
        return RankedResult(kept=(), dropped=tuple(ordered), primary=UNKNOWN_INTENT,
                            multi_intent=False, confidence=0.0,
                            reason=_LOW_CONFIDENCE)
    kept = tuple(_intent_match(score) for score in kept_scores)
    below_cap = sorted(candidates[len(kept_scores):] + rejected, key=_ranking_key)
    return RankedResult(kept=kept, dropped=tuple(below_cap),
                        primary=kept[0].intent, multi_intent=len(kept) > 1,
                        confidence=confidence, reason=_OK)


def _ranking_key(score: IntentScore) -> tuple[float, int, str]:
    """A chave da ordem total: ponto decrescente, prioridade decrescente, id
    crescente.

    Os dois primeiros termos saem com o sinal trocado porque `sorted` ordena
    crescente e a ordem é "melhor primeiro". O terceiro fica sem sinal porque é
    o desempate do desempate e o desempate vai para o fim dos dois: com o
    `intent_id` ele fecha a conta, e como o `RuleSet` recusa id repetido (§4.5
    do spec) duas intenções distintas nunca empatam nas três chaves — a ordem é
    total mesmo, e não depende de como o `dict` de alguém iterou.

    O `intent_id` também é o que impede a lista de diagnóstico de mudar de
    ordem quando o `RuleSet` muda de versão: `evaluate` devolve na ordem do
    arquivo, e a ordem do arquivo é conteúdo.
    """
    return (-score.score, -score.priority, score.intent_id)


def _clears_floor(score: IntentScore) -> bool:
    """True quando a intenção passa do seu próprio `min_score` com ponto acima
    de zero.

    As duas condições são o mesmo piso dito de dois jeitos, e as duas são
    necessárias. `score >= min_score` é §4.4 do spec: o `min_score` é o que
    quem escreve o arquivo de regras pediu, e ele vem junto na `IntentScore`, do
    `RuleSet`, sem passar pelo `Settings`. `score > 0.0` é a negação do Task 6
    chegando aqui: uma intenção totalmente negada volta com ponto zero e com o
    trace cheio de acertos marcados, e tratá-la como candidata colocaria na
    resposta uma evidência que o próprio motor recuou.

    Com um `RuleSet` validado a segunda condição nunca decide nada sozinha —
    `rules.py` recusa `min_score <= 0` no carregamento —, e é por isso que ela é
    uma guarda e não um segundo filtro: ela existe para o `IntentScore` que um
    teste monta à mão, onde o `min_score` pode ser zero, e para o dia em que
    alguém relaxar o validador sem perceber o que abria.
    """
    return score.score > 0.0 and score.score >= score.min_score


def _intent_match(score: IntentScore) -> IntentMatch:
    """A `IntentMatch` de uma intenção que sobreviveu ao filtro e ao teto.

    `score` e `min_score` vão crus, e a confiança vai arredondada: quem lê a
    tela precisa do `score / min_score` da intenção que está olhando, e
    arredondar o denominadoraria a comparação que ela está tentando fazer. O
    `rule_hits` é o trace inteiro, incluindo o que a negação zerou — a negação
    é visível aqui justamente porque §5.2 manda, e é essa visibilidade que
    distingue "o motor errou" de "o motor decidiu certo e mostrou por quê".
    """
    return IntentMatch(
        intent=score.intent_id,
        label=score.label,
        score=score.score,
        confidence=confidence_for(score.score),
        priority=score.priority,
        min_score=score.min_score,
        rule_hits=tuple(_rule_view(hit) for hit in score.hits),
    )


def _rule_view(hit: RawHit) -> RuleView:
    """O `RuleView` de um acerto, com os deslocamentos já no texto original.

    `type` sai como `hit.type.value` e não como o membro do `StrEnum`: o campo
    de `RuleView` é `str` e a API serializa o resultado, e o valor é a palavra
    que o autor da regra escreveu no arquivo — `"keyword"`, `"pattern"`. O resto
    é cópia literal: `matched_text` já vem do `Doc` sobre o texto do usuário, e
    `offsets_exact` viaja porque é ele que avisa que o começo mostrado não é o
    começo que o padrão cobriu.
    """
    return RuleView(
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


@dataclass(frozen=True)
class _Negation:
    """A janela de negação de um idioma, já montada: os negadores do arquivo de
    regras agrupados por número de palavras, do mais longo para o mais curto, as
    palavras que fecham uma oração, e quantos tokens a varredura percorre.

    Agrupar por comprimento é o que permite que `deixar de` seja um negador
    inteiro: a comparação é de token para token, e `deixar` sozinho não está na
    lista. A ordem é do mais longo para o mais curto porque `não quero` e `não`
    podem estar na lista ao mesmo tempo e, na mesma posição, o negador mais longo
    é o que descreve melhor o que o usuário escreveu.
    """

    by_length: tuple[tuple[int, frozenset[tuple[str, ...]]], ...]
    boundaries: frozenset[str]
    window: int

    def negates(self, lexical: _Lexical, hit: RawHit) -> bool:
        """True quando um negador está na janela à esquerda de `hit`.

        A varredura anda pela sequência lexical de `lexical`, e a janela conta
        passos nela: um espaço em branco não é palavra, não gasta passo e não
        separa as palavras de um negador. A forma comparada é a mesma das
        fronteiras e dos negadores, para que a comparação não dependa de acento
        nem de caixa em nenhum dos dois lados. A varredura parte do primeiro token
        do acerto, não do primeiro token da regra: no tipo `all` cada acerto é de
        uma palavra só, e é essa palavra que pode estar negada.

        Um acerto sem token não é negável. Sem posição não há o que varrer, e um
        `RawHit` só fica sem token quando um padrão cobriu um trecho que nenhum
        token do documento cobre.
        """
        if not hit.token_indices:
            return False
        start = lexical.at_or_after(min(hit.token_indices))
        if start is None:
            return False
        forms = lexical.forms
        position = start - 1
        for _ in range(self.window):
            if position < 0:
                return False
            form = forms[position]
            # o negador antes da fronteira, e a ordem é o contrato: uma palavra de
            # fronteira que também é a última de um negador de duas palavras
            # ("não quero", com `e` na lista) precisa negar, e invertendo a ordem a
            # varredura pararia nela e a negação se perderia
            if self._is_negator(forms, position):
                return True
            if _is_clause_break(form):
                return False
            if form in self.boundaries:
                return False
            position -= 1
        return False

    def _is_negator(self, forms: tuple[str, ...], position: int) -> bool:
        """True quando algum negador termina em `position`.

        Os `length` tokens que terminam em `position` viram uma tupla e são
        procurados no grupo daquele comprimento; um grupo que nem cabe à esquerda
        de `position` é pulado, porque uma fatia com início negativo contornaria
        o fim do documento em vez de mostrar que o texto acabou.
        """
        for length, phrases in self.by_length:
            start = position - length + 1
            if start >= 0 and tuple(forms[start:position + 1]) in phrases:
                return True
        return False


def _build_negation(negators: tuple[str, ...], boundaries: frozenset[str],
                    window: int) -> _Negation:
    """Monta a tabela da janela a partir do que o arquivo de regras carregou.

    As três entradas são normalizadas com a mesma função usada nos tokens, e não
    em `rules.py`: quem escreve o arquivo escreve `não`, `porém` e `Não` como se
    fossem a mesma coisa, e a janela de negação é o único lugar onde essa
    diferença é resolvida. O `RuleSet` guarda as frases como vieram do JSON — é
    conteúdo, e conteúdo se mostra como foi digitado.
    """
    grouped: dict[int, set[tuple[str, ...]]] = {}
    for negator in negators:
        words = tuple(_negation_form(word) for word in negator.split())
        grouped.setdefault(len(words), set()).add(words)
    by_length = tuple(
        (length, frozenset(grouped[length]))
        for length in sorted(grouped, reverse=True)
    )
    return _Negation(
        by_length=by_length,
        boundaries=frozenset(_negation_form(word) for word in boundaries),
        window=window,
    )


def _surface_form(token: Token) -> str:
    """A forma de superfície do token, em minúsculas."""
    return token.lower_


def _lemma_form(token: Token) -> str:
    """O lema do token, em minúsculas."""
    return token.lemma_.lower()


def _negation_form(text: str) -> str:
    """O texto na forma em que a janela de negação compara dos dois lados.

    Minúsculas primeiro e acento depois, e não o contrário: `İ` perde o acento
    combinante só depois de descer, e quem o escreve é o tokenizer do spaCy sobre
    uma mensagem em qualquer idioma. A mesma função entra nos negadores, nas
    fronteiras e nos tokens, e é por isso que a comparação não precisa de acento
    nem de caixa em nenhum dos três.
    """
    return strip_accents(text.lower())


def _is_clause_break(form: str) -> bool:
    """True quando a forma fecha uma oração: pontuação ou quebra de linha.

    A lista é do código e não do arquivo de regras, e é o lugar certo dela: quem
    escreve o arquivo de regras não tem como saber que a vírgula de
    `negation_boundaries` é o que impede a negação de atravessar a oração, e um
    arquivo que a esquecesse devolveria "não gostei, mas quero cancelar" como
    `cancelar` negado, sem nada reclamar.

    A lista é fechada, e é por isso que a quebra de linha é testada por caractere
    e não por `isspace`. O tokenizer do spaCy entrega um token para qualquer
    sequência de dois ou mais espaços em branco, então `isspace` também parava a
    negação num espaço duplo ou num tabulador: "não  quero cancelar" voltava
    como `cancelar` confiante. O usuário digita espaço duplo sem perceber, e uma
    negação perdida assim é a pior falha possível nesta feature - um erro
    silencioso, na resposta que a tela mostra. `\r\n` cai no teste do `\n`, e
    `\n`, `\n\n` e um `\n` com espaço ao lado caem todos no mesmo lugar.
    """
    return form in _CLAUSE_BREAKS or "\n" in form or "\r" in form


def _is_skippable(form: str) -> bool:
    """True quando o token é espaço em branco e não é quebra de linha.

    A stance do motor sobre espaço em branco é uma só, e é a mesma aqui e em
    `normalize.py`: ele é colapsado e não significa nada, então não é evidência.
    Um espaço duplo ou um tabulador é o teclado do usuário, não uma fronteira de
    oração, e o tokenizer do spaCy entrega um token próprio para qualquer
    sequência de dois ou mais. Deixar esse token na varredura faz ele gastar um
    passo da janela e abrir um corte no meio de um negador de duas palavras, e
    aí "deixar  de cancelar" volta como `cancelar` com o peso inteiro.

    A exceção é a quebra de linha, e ela é a exceção porque é a forma de a
    mensagem chegar estruturada: uma linha nova é uma oração nova, ou uma
    mensagem nova. Ela é espaço em branco como token, então ficaria pulada junto
    com o resto, e é por isso que o teste é o mesmo de `_is_clause_break` —
    `not _is_clause_break(form)`, que para uma forma de espaço em branco se
    resume ao `\n` e ao `\r` sem repetir a lista aqui.
    """
    return form.isspace() and not _is_clause_break(form)


@dataclass(frozen=True)
class _Lexical:
    """Os tokens lexurais da mensagem, na ordem, com o índice de cada um no `Doc`.

    A janela de negação caminha por esta sequência e não pelas posições do `Doc`.
    Um token que não é palavra não pode gastar um passo da janela nem partir um
    negador de duas palavras ao meio, e o `Doc` é o único lugar onde esses tokens
    existem. `doc_indices` é a ponte nos dois sentidos: `token_indices` continua em
    índices do `Doc`, que é o que `Task 5` gravou e o que a regra de negação tem de
    devolver, e `at_or_after` traduz um desses índices para a posição daqui.

    Uma quebra de linha é um token só de espaços em branco e mesmo assim fica na
    sequência, porque `_is_clause_break` é que a para, e é assim que a exceção de
    `_is_skippable` se aplica sem um segundo mecanismo.
    """

    forms: tuple[str, ...]
    doc_indices: tuple[int, ...]

    @classmethod
    def build(cls, forms: tuple[str, ...]) -> _Lexical:
        """Monta a sequência tirando os tokens que `_is_skippable` dispensa."""
        kept = [(form, index) for index, form in enumerate(forms)
                if not _is_skippable(form)]
        return cls(tuple(form for form, _ in kept),
                   tuple(index for _, index in kept))

    def at_or_after(self, doc_index: int) -> int | None:
        """A posição do primeiro token lexical em `doc_index` ou depois dele.

        "Ou depois" é o que faz o padrão de `Task 5` que começa por um espaço em
        branco continuar certo: ele pode abrir num token de espaço, e o primeiro
        token lexical do acerto é a palavra logo depois dele. `None` quando não
        sobra token nenhum, e aí não há o que varrer.
        """
        position = bisect_left(self.doc_indices, doc_index)
        return None if position == len(self.forms) else position


class _IntentEvaluation:
    """As duas passagens de uma intenção, a negação, e a soma do peso por regra.

    O estado é todo de uma chamada: `consumed` é o que a primeira passagem
    tomou para si, `spans` e `claimed` são a deduplicação, e `credited` é o
    que já entrou na soma. Nenhum deles sobrevive à avaliação — e `forms`, a forma
    normalizada de cada token, é da mensagem e não da intenção, porque é a mesma
    para todas elas e refazê-la por intenção seria trabalho jogado fora.
    """

    def __init__(self, doc: Doc, normalised: Normalised, intent: IntentSpec,
                 negation: _Negation, lexical: _Lexical) -> None:
        self._doc = doc
        self._normalised = normalised
        self._intent = intent
        self._negation = negation
        self._lexical = lexical
        self._consumed: set[int] = set()
        self._spans: set[tuple[str, int, int]] = set()
        self._claimed: set[tuple[str, int]] = set()
        self._credited: dict[str, float] = {}
        self.hits: list[RawHit] = []
        self.score = 0.0

    def run(self) -> None:
        """Roda as duas passagens, nesta ordem: `all` primeiro, o resto depois.

        A negação é a última, porque só depois das duas há uma lista de acertos
        para varrer: no meio delas um `all` ainda pode não ter as duas palavras, e
        um padrão que casou por cima de um token consumido só é descartado na
        segunda passagem.
        """
        for rule in self._intent.rules:
            if rule.type is RuleType.ALL:
                self._run_all(rule)
        for rule in self._intent.rules:
            if rule.type is not RuleType.ALL:
                self._run_single(rule)
        self._apply_negation()

    def _run_all(self, rule: Rule) -> None:
        """Primeira passagem: a conjunção e o consumo dos tokens dela.

        A regra exige *todos* os valores, então a busca só é consumada depois
        que cada palavra foi encontrada: um `all` que casa metade não pode
        deixar a outra metade reservada para uma regra que vem depois. A união
        dos tokens entra inteira em `consumed` — o tipo `all` serve para exigir
        sinais independentes, e a segunda menção da mesma palavra é o mesmo
        sinal. O trace, esse, mostra uma ocorrência por palavra, a primeira de
        cada uma: `token_indices` de um acerto aponta para o token que ele
        próprio casou, que é de onde a negação parte.
        """
        found: list[tuple[str, list[Token]]] = []
        for value in rule.value:
            matches = self._matching_tokens(value)
            if not matches:
                return
            found.append((value, matches))
        for _, matches in found:
            self._consumed.update(token.i for token in matches)
        for value, matches in found:
            self._record(self._token_hit(rule, value, matches[0]))

    def _run_single(self, rule: Rule) -> None:
        """Segunda passagem: uma regra de token único, ou um `pattern`.

        Pula o que a primeira passagem consumiu, e só isso: entre uma regra
        `keyword` e outra, o consumo não arbitra, senão a pontuação voltaria a
        depender da ordem do arquivo.
        """
        if rule.type is RuleType.PATTERN:
            for hit in self._pattern_hits(rule):
                if self._touches_consumed(hit.token_indices):
                    continue
                self._record(hit)
            return
        if rule.type is RuleType.LEMMA and not self._doc.has_annotation("LEMMA"):
            # o pipe ativo não lematiza: a regra é descartada em silêncio e o
            # `ResourceReport` diz que ela não está sendo usada. Um `blank` sem
            # modelo é o caso comum, não uma falha do pedido.
            return
        value = rule.value
        if not isinstance(value, str):
            return
        form_of = _lemma_form if rule.type is RuleType.LEMMA else _surface_form
        for token in self._matching_tokens(value, form_of):
            if token.i in self._consumed:
                continue
            self._record(self._token_hit(rule, value, token))

    def _pattern_hits(self, rule: Rule) -> list[RawHit]:
        """Os acertos de um `pattern`, em ordem de documento.

        Casa primeiro contra o texto normalizado, e depois contra a cópia sem
        pontuação: pessoa digita `cancelar, meu plano`, e uma regra de frase
        escrita `cancelar\\s+(meu\\s+)?plano` existe para pegar a frase, não
        para punir a vírgula. A cópia tem exatamente o mesmo comprimento — cada
        caractere de pontuação vira um espaço, um por um —, então o mapa de
        índices de `normalise` continua valendo e nenhum deslocamento muda por
        causa dessa tolerância. O que se perde é a possibilidade de escrever um
        padrão que *exija* um caractere de pontuação; quem precisar do fim de
        frase usa a pontuação como fronteira, que é o que o padrão implícito
        `(?<!\\w)…(?!\\w)` já garante.
        """
        hits = [
            self._pattern_hit(rule, *match.span())
            for text in self._matchable_texts()
            for match in rule.compiled.finditer(text)
            if match.start() != match.end()
        ]
        # a cópia tolerante pode ter achado uma ocorrência que a passagem
        # principal não achou, e ela vem antes no texto
        return sorted(hits, key=lambda hit: (hit.char_start, hit.char_end))

    def _pattern_hit(self, rule: Rule, start: int, end: int) -> RawHit:
        """Um acerto de `pattern`, com `start` e `end` no texto normalizado.

        A translate do fim passa por `to_original_offset` porque o normalizado é
        mais curto que o original sempre que há acento: `não` vira `nao`, e
        tudo depois disso anda um caractere para a esquerda. O token que o
        padrão cobre é o que se sobrepõe ao trecho já traduzido, e é por esse
        token que o acerto aparece na tabela de tokens.
        """
        normalised = self._normalised
        char_start = to_original_offset(normalised, start)
        char_end = to_original_offset(normalised, end)
        offsets_exact = _boundaries_are_exact(normalised, start, end)
        token_indices = tuple(
            token.i for token in self._doc
            if token.idx < char_end and token.idx + len(token.text) > char_start
        )
        if token_indices and self._doc[token_indices[0]].idx != char_start:
            # o padrão não começou onde o primeiro token que ele cobre começa:
            # entrou no meio de um token (`50` dentro de `-50`, que o tokenizer
            # do spaCy entrega como um token só) ou no espaço que separa um
            # token do outro (`\\s+maior`). O acerto é reportado pelo token,
            # porque é o token que a interface mostra e é o token que a negação
            # percorre, e `offsets_exact` avisa que o começo mostrado não é o
            # começo que o padrão cobriu
            char_start = self._doc[token_indices[0]].idx
            offsets_exact = False
        return RawHit(
            rule_id=rule.rule_id,
            type=rule.type,
            value=rule.value,
            weight=rule.weight,
            char_start=char_start,
            char_end=char_end,
            matched_text=self._doc.text[char_start:char_end],
            offsets_exact=offsets_exact,
            token_indices=token_indices,
        )

    def _touches_consumed(self, token_indices: Iterable[int]) -> bool:
        """True quando o acerto cobre algum token que um `all` já consumiu.

        O acerto inteiro some, e não só o token repetido: um `pattern` que
        passa por cima de uma palavra que a conjunção já pagou não é evidência
        nova, e sobrar dele um token qualquer do trecho faria o acerto
        parecer mais estreito do que é — `cancelar\\s+a\\s+assinatura` sobre
        `cancelar a assinatura` casaria pelo artigo e pontuaria de novo pela
        mesma frase.
        """
        return any(index in self._consumed for index in token_indices)

    def _matchable_texts(self) -> tuple[str, ...]:
        """O texto normalizado, e a cópia sem pontuação quando ela muda algo."""
        text = self._normalised.text
        flattened = _without_punctuation(text)
        return (text,) if flattened == text else (text, flattened)

    def _matching_tokens(
        self, value: str, form_of: Callable[[Token], str] = _surface_form
    ) -> list[Token]:
        """Os tokens cuja forma casa com `value`, na ordem do documento."""
        expected, stripped = _expected_forms(value)
        return [token for token in self._doc
                if _form_matches(form_of(token), expected, stripped)]

    def _token_hit(self, rule: Rule, value: str, token: Token) -> RawHit:
        """O acerto de um token: os deslocamentos são os do próprio token.

        Um token é um trecho real do texto que o usuário digitou, então aqui
        `offsets_exact` é sempre `True`: só a tradução de um `pattern` pode
        perder a conta.
        """
        start = token.idx
        end = start + len(token.text)
        return RawHit(
            rule_id=rule.rule_id,
            type=rule.type,
            value=value,
            weight=rule.weight,
            char_start=start,
            char_end=end,
            matched_text=self._doc.text[start:end],
            offsets_exact=True,
            token_indices=(token.i,),
        )

    def _record(self, hit: RawHit) -> bool:
        """Guarda o acerto, se ele for evidência nova da sua regra.

        Duas recusas, e as duas são a mesma pergunta — "esta regra já contou
        isto?": pelo trecho, porque o mesmo padrão casa o mesmo trecho nas duas
        passagens de texto; e pelo par `(regra, token)`, porque um token só
        pode ser contado uma vez pela mesma regra, ainda que dois acertos
        diferentes o tocem. O peso entra na soma na primeira vez, e só nela.
        """
        if (hit.rule_id, hit.char_start, hit.char_end) in self._spans:
            return False
        if any((hit.rule_id, index) in self._claimed for index in hit.token_indices):
            return False
        self._spans.add((hit.rule_id, hit.char_start, hit.char_end))
        self._claimed.update((hit.rule_id, index) for index in hit.token_indices)
        self.hits.append(hit)
        if hit.rule_id not in self._credited:
            self._credited[hit.rule_id] = hit.weight
            self.score += hit.weight
        return True

    def _apply_negation(self) -> None:
        """Marca `negated` em cada acerto e refaz a soma do peso.

        A soma é refeita, e não subtraída, para a ordem da adição continuar sendo
        a ordem em que as regras entraram: a expressão é a mesma de antes, sem
        depender de a subtração cair na casa certa.

        Duas perguntas sobre o peso, e a resposta é outra para cada tipo de regra.
        Para `keyword`, `lemma` e `pattern` a pergunta é "sobrou alguma
        ocorrência?": a mesma palavra dita duas vezes são duas afirmações, e
        negar uma não diz nada sobre a outra. Para `all` a pergunta é
        "sobrou *todo* o conjunto?": `all ["cancelar","assinatura"]` não é
        "cancelar" e "assinatura" para o motor, é uma afirmação só escrita como
        conjunção, e "não cancelar, mas assinatura" não afirma essa conjunção.
        Com 3.5 ela passava do `min_score` de 3.0 da própria intenção, e o
        resultado era classificar como `cancelar` a frase que alguém digita
        justamente para conferir que funciona.
        """
        for hit in self.hits:
            hit.negated = self._negation.negates(self._lexical, hit)
        surviving: set[str] = set()
        withdrawn: set[str] = set()
        for hit in self.hits:
            if not hit.negated:
                surviving.add(hit.rule_id)
            elif hit.type is RuleType.ALL:
                withdrawn.add(hit.rule_id)
        self.score = sum(self._credited[rule_id] for rule_id in self._credited
                         if rule_id in surviving and rule_id not in withdrawn)




def _expected_forms(value: str) -> tuple[str, str]:
    """O valor da regra em minúsculas, e o mesmo valor sem acento.

    As duas formas porque o `Doc` foi construído sobre o texto que o usuário
    digitou, com os acentos que ele digitou, e o valor da regra foi escrito sem
    pensar em acento: `nao` tem que casar com `não`. A comparação é de token
    inteiro nos dois casos, então nenhuma das duas formas cria casamento dentro
    de palavra maior.
    """
    lowered = value.lower()
    return lowered, strip_accents(lowered)


def _form_matches(form: str, expected: str, stripped: str) -> bool:
    """True quando a forma do token é o valor da regra, com ou sem acento."""
    return form == expected or strip_accents(form) == stripped


def _boundaries_are_exact(normalised: Normalised, start: int, end: int) -> bool:
    """True quando as duas bordas do acerto traduzem para o original sem perda.

    Perde quando a borda cai sobre um espaço que `normalise` sintetizou: uma
    sequência de espaços no original vira um só no texto normalizado, e esse
    espaço guarda o índice do *começo* da sequência, então um padrão que
    termina sobre ele aponta para depois dos espaços, não sobre eles. O trace
    diz `offsets_exact = False` em vez de destacar um trecho torto. Uma borda
    que cai no fim do texto não conta como perda: `to_original_offset` devolve
    o índice do último caractere mais um, que é o fim do que o padrão cobriu.
    """
    return not (_collapsed_space(normalised, start)
                or _collapsed_space(normalised, end - 1))


def _collapsed_space(normalised: Normalised, offset: int) -> bool:
    """True quando `offset` cai sobre um espaço que a normalização juntou."""
    if not 0 <= offset < len(normalised.text) or normalised.text[offset] != " ":
        return False
    following = offset + 1
    if following >= len(normalised.index_map):
        return False
    return normalised.index_map[following] - normalised.index_map[offset] > 1


def _without_punctuation(text: str) -> str:
    """A mesma string com a pontuação trocada por espaços, um por um.

    O comprimento não muda, e é esse o ponto: o mapa de índices de `normalise`
    é posicional, então trocar um caractere por outro não move nada. A troca é
    por caractere e nunca por trecho, para que um `\\s+` do padrão continue
    Vendo exatamente um separador entre duas palavras.
    """
    return "".join(char if char.isalnum() or char.isspace() else " " for char in text)

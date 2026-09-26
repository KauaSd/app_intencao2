"""Chatbot: rule-based, one strategy per intent. No generation."""

import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from intent.pipeline import Pipeline
from intent.schemas import IntentMatch, IntentResult, LanguageInfo


SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    re.compile(r"(?i)minha\s+senha\s+(?:é|eh)\s+\S+"),
    re.compile(r"(?i)bearer\s+[a-z0-9\-._~+/]+=*"),
]


def redact(text: str) -> str:
    """Redact sensitive patterns from text before storage/echo."""
    result = text
    for pattern in SENSITIVE_PATTERNS:
        result = pattern.sub("[REDACTADO]", result)
    return result


@dataclass(frozen=True)
class ChatResponse:
    """Response from the chatbot."""
    reply: str
    reply_format: str
    session_id: str
    intents: tuple[IntentMatch, ...]
    primary: str
    multi_intent: bool
    strategy: str
    confidence: float
    turn_index: int
    pending_step: Optional[int] = None

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    """Chat session state."""
    session_id: str
    turns: list[dict] = field(default_factory=list)
    pending_intent: Optional[str] = None
    pending_step: int = 0
    collected: dict = field(default_factory=dict)

    def add_turn(self, role: str, text: str, intents: tuple[IntentMatch, ...] = (),
                 primary: str = "", strategy: str = "", pending_step: Optional[int] = None) -> None:
        self.turns.append({
            "role": role,
            "text": text,
            "intents": intents,
            "primary": primary,
            "strategy": strategy,
            "pending_step": pending_step,
        })


class Chatbot:
    """Rule-based chatbot with one strategy per intent."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def _get_or_create_session(self, session_id: Optional[str]) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        new_id = session_id or str(uuid.uuid4())[:8]
        session = Session(session_id=new_id)
        self._sessions[new_id] = session
        return session

    def responder(
        self,
        mensagem: str,
        session_id: Optional[str] = None,
        pipeline: Optional[Pipeline] = None,
    ) -> ChatResponse:
        """Process a user message and return a chatbot response."""
        session = self._get_or_create_session(session_id)

        mensagem = redact(mensagem)

        detected_result: Optional[IntentResult] = None
        if pipeline:
            detected_result = pipeline.detect(mensagem)

        primary = detected_result.primary if detected_result else "desconhecido"
        intents = detected_result.intents if detected_result else ()
        multi_intent = detected_result.multi_intent if detected_result else False
        confidence = detected_result.confidence if detected_result else 0.0

        turn_index = len(session.turns) + 1

        strategy, reply, reply_format, pending_step = self._execute_strategy(
            primary, intents, session, mensagem, confidence
        )

        session.add_turn(
            role="user",
            text=mensagem,
            intents=intents,
            primary=primary,
            strategy=strategy,
            pending_step=pending_step,
        )

        response = ChatResponse(
            reply=reply,
            reply_format=reply_format,
            session_id=session.session_id,
            intents=intents,
            primary=primary,
            multi_intent=multi_intent,
            strategy=strategy,
            confidence=confidence,
            turn_index=turn_index,
            pending_step=pending_step,
        )

        session.add_turn(
            role="bot",
            text=reply,
            intents=intents,
            primary=primary,
            strategy=strategy,
            pending_step=pending_step,
        )

        return response

    def _execute_strategy(
        self,
        primary: str,
        intents: tuple[IntentMatch, ...],
        session: Session,
        mensagem: str,
        confidence: float,
    ) -> tuple[str, str, str, Optional[int]]:
        """Execute the strategy for the detected intent."""

        if session.pending_intent and primary != "desconhecido":
            if primary != session.pending_intent and primary in self._collector_intents():
                session.pending_intent = None
                session.pending_step = 0
                session.collected = {}

        if session.pending_intent:
            if primary == "desconhecido" or confidence < 0.3:
                return self._advance_collector(session.pending_intent, session, mensagem)
            if primary != session.pending_intent and primary in self._collector_intents():
                session.pending_intent = None
                session.pending_step = 0
                session.collected = {}

        if primary == "saudacao":
            return "saudacao", self._strategy_saudacao(), "text", None

        if primary == "cancelar":
            return self._handle_collector("cancelar", session, mensagem, 3,
                ["Confirme: deseja cancelar sua assinatura? (sim/não)",
                 "Qual o motivo do cancelamento?",
                 "Seu protocolo de cancelamento é: CANC-{:04d}. Processado."])

        if primary == "comprar":
            return "comprar", self._strategy_comprar(), "text", None

        if primary == "upgrade":
            return "upgrade", self._strategy_upgrade(), "text", None

        if primary == "downgrade":
            return "downgrade", self._strategy_downgrade(), "text", None

        if primary == "suporte":
            return "suporte", self._strategy_suporte(mensagem), "text", None

        if primary == "acesso":
            return self._handle_collector("acesso", session, mensagem, 3,
                ["Informe seu e-mail ou usuário para iniciar a recuperação.",
                 "Enviamos um link de recuperação para seu e-mail. Verifique a caixa de entrada.",
                 "Sua senha foi redefinida. Novo acesso liberado."],
                never_ask_password=True)

        if primary == "pagamento":
            return "pagamento", self._strategy_pagamento(mensagem), "text", None

        if primary == "alterar_dados":
            return self._handle_collector("alterar_dados", session, mensagem, 2,
                ["Qual dado deseja alterar? (nome, e-mail, endereço, documento)",
                 "Dado alterado com sucesso."])

        if primary == "reclamacao":
            return "reclamacao", self._strategy_reclamacao(), "text", None

        if primary == "duvida":
            return "duvida", self._strategy_duvida(mensagem), "text", None

        return "desconhecido", self._strategy_desconhecido(mensagem), "text", None

    def _collector_intents(self) -> set[str]:
        return {"cancelar", "acesso", "alterar_dados"}

    def _handle_collector(
        self,
        intent: str,
        session: Session,
        mensagem: str,
        steps: int,
        step_messages: list[str],
        never_ask_password: bool = False,
    ) -> tuple[str, str, str, Optional[int]]:
        """Handle multi-step collector flows."""
        if session.pending_intent != intent:
            session.pending_intent = intent
            session.pending_step = 1
            session.collected = {}
            return intent, step_messages[0], "text", 1

        session.pending_step += 1
        step = session.pending_step

        if step <= len(step_messages):
            reply = step_messages[step - 1]
            if "{:04d}" in reply:
                import random
                reply = reply.format(random.randint(1000, 9999))
        else:
            reply = "Processo concluído."

        if step >= steps:
            session.pending_intent = None
            session.pending_step = 0
            session.collected = {}
            return intent, reply, "text", None

        return intent, reply, "text", step

    def _strategy_saudacao(self) -> str:
        return ("Olá! Como posso ajudar hoje?\n"
                "Opções: cancelar, comprar, suporte, upgrade, downgrade, "
                "acesso, pagamento, alterar dados, reclamação, dúvida.")

    def _strategy_comprar(self) -> str:
        return ("Temos os planos: Básico (R$ 29/mês), Pro (R$ 79/mês), "
                "Enterprise (sob consulta). Qual plano deseja assinar?")

    def _strategy_upgrade(self) -> str:
        return ("Você está no plano Básico. Deseja migrar para Pro (R$ 79/mês) "
                "ou Enterprise (sob consulta)?")

    def _strategy_downgrade(self) -> str:
        return ("Deseja migrar para o plano Básico (R$ 29/mês)? "
                "A mudança valerá no próximo ciclo de cobrança.")

    def _strategy_suporte(self, mensagem: str) -> str:
        ticket_id = f"SUP-{abs(hash(mensagem)) % 10000:04d}"
        if any(w in mensagem.lower() for w in ["erro", "não carrega", "nao carrega", "não abre", "nao abre"]):
            return f"Detectei um problema técnico. Seu ticket: {ticket_id}. Nossa equipe vai investigar."
        return f"Suporte acionado. Ticket: {ticket_id}. Em breve entraremos em contato."

    def _strategy_pagamento(self, mensagem: str) -> str:
        msg_lower = mensagem.lower()
        if "duas vezes" in msg_lower or "duplicado" in msg_lower or "cobrou" in msg_lower:
            return ("Identifiquei cobrança duplicada. Iniciando estorno automático. "
                    "O valor retorna em até 5 dias úteis.")
        return ("Para pagamento: PIX, boleto ou cartão. Qual método prefere? "
                "Se já pagou, envie o comprovante para agilizar.")

    def _strategy_reclamacao(self) -> str:
        return ("Registramos sua reclamação e abrimos um protocolo. "
                "Um especialista entrará em contato em até 24h. "
                "Deseja falar com um atendente agora?")

    def _strategy_duvida(self, mensagem: str) -> str:
        faq = {
            "preço": "Planos a partir de R$ 29/mês. Veja detalhes em /comprar.",
            "teste": "Temos 7 dias grátis no plano Pro. Sem compromisso.",
            "cancelar": "Pode cancelar a qualquer momento. Sem multa.",
        }
        msg_lower = mensagem.lower()
        for key, answer in faq.items():
            if key in msg_lower:
                return answer
        return ("Não tenho essa informação no momento. "
                "Posso transferir para um atendente humano?")

    def _strategy_desconhecido(self, mensagem: str) -> str:
        return ("Não entendi sua mensagem. Vou transferir para um atendente humano. "
                f"Sua mensagem original: \"{mensagem}\"")
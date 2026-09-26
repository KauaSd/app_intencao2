"""Typer CLI for the intent engine."""

import json
import sys
from pathlib import Path

import typer

from intent.config import Settings
from intent.language import load_profiles
from intent.pipeline import Pipeline
from intent.rules import load_ruleset, load_all_rulesets, RuleSet

app = typer.Typer(
    name="intent",
    help="Motor determinístico de detecção de intenção",
    no_args_is_help=True,
)

settings = Settings.from_env()
rulesets = load_all_rulesets(settings.languages)
profiles = load_profiles()
pipeline = Pipeline(rulesets=rulesets, profiles=profiles, settings=settings)


@app.command()
def detect(
    text: str = typer.Argument(None, help="Mensagem para classificar"),
    json_output: bool = typer.Option(False, "--json", help="Saída em JSON"),
    stdin: bool = typer.Option(False, "--stdin", help="Ler uma mensagem por linha do stdin"),
) -> None:
    """Detecta a intenção de uma mensagem."""
    if stdin:
        lines = [line.strip() for line in sys.stdin if line.strip()]
        if not lines:
            return
        for line in lines:
            result = pipeline.detect(line)
            _print_result(result, json_output)
        return

    if not text:
        typer.echo("Erro: mensagem obrigatória (ou use --stdin)", err=True)
        raise typer.Exit(2)

    result = pipeline.detect(text)
    _print_result(result, json_output)
    if result.primary == "desconhecido":
        raise typer.Exit(1)


@app.command()
def batch(file: Path = typer.Argument(..., help="Arquivo JSON com array de mensagens")) -> None:
    """Processa um lote de mensagens de um arquivo JSON."""
    with file.open(encoding="utf-8") as f:
        messages = json.load(f)
    if not isinstance(messages, list):
        typer.echo("Erro: arquivo deve conter um array JSON", err=True)
        raise typer.Exit(2)

    results = pipeline.detect_batch(messages)
    for result in results:
        _print_result(result, json_output=True)


@app.command()
def list() -> None:
    """Lista o catálogo de intenções com contagem de regras."""
    for lang, ruleset in rulesets.items():
        typer.echo(f"\n=== {lang.upper()} ===")
        for intent in ruleset.intents:
            typer.echo(f"  {intent.id:20} {intent.label}  ({len(intent.rules)} regras)")


@app.command(name="rules")
def rules_cmd(
    show: str = typer.Option(None, "--show", help="Mostrar regras de um idioma"),
    validate: Path = typer.Option(None, "--validate", help="Validar arquivo de regras"),
) -> None:
    """Comandos relacionados a regras."""
    if show:
        if show not in rulesets:
            typer.echo(f"Erro: idioma '{show}' não encontrado", err=True)
            raise typer.Exit(3)
        ruleset = rulesets[show]
        for intent in ruleset.intents:
            typer.echo(f"\n{intent.id} ({intent.label})")
            for rule in intent.rules:
                typer.echo(f"  {rule.type.value:8} {rule.value!r}  weight={rule.weight}")

    if validate:
        try:
            load_ruleset("pt", path=validate)
            typer.echo(f"OK: {validate} válido")
        except Exception as e:
            typer.echo(f"Erro: {e}", err=True)
            raise typer.Exit(3)


@app.command()
def demo() -> None:
    """Executa exemplos embutidos por intenção."""
    demo_cases = [
        ("saudacao", "oi, tudo bem?"),
        ("saudacao", "bom dia"),
        ("cancelar", "quero cancelar meu plano"),
        ("cancelar", "como faço para cancelar a assinatura?"),
        ("comprar", "quero assinar o plano pro"),
        ("comprar", "gostaria de contratar um plano"),
        ("upgrade", "quero um plano maior"),
        ("downgrade", "quero baixar meu plano"),
        ("suporte", "a pagina nao carrega"),
        ("suporte", "preciso de suporte tecnico"),
        ("acesso", "esqueci minha senha"),
        ("acesso", "conta bloqueada"),
        ("pagamento", "cobraram duas vezes"),
        ("pagamento", "quero o boleto"),
        ("alterar_dados", "preciso mudar meu email"),
        ("alterar_dados", "atualizar endereco"),
        ("reclamacao", "nao aceito isso, quero falar com gerente"),
        ("reclamacao", "isso e um absurdo, vou processar"),
        ("duvida", "quanto custa o plano basico?"),
        ("duvida", "como funciona o periodo de teste?"),
    ]

    for expected, text in demo_cases:
        result = pipeline.detect(text)
        status = "OK" if result.primary == expected else "FAIL"
        typer.echo(f"  {status}  {text!r} -> {result.primary} (esperado: {expected})")


@app.command()
def chat(session_id: str = typer.Option(None, "--session", help="ID da sessão")) -> None:
    """Loop interativo do chatbot."""
    from intent.chat import Chatbot
    from intent.service import ChatService

    chatbot = Chatbot()
    service = ChatService(pipeline, chatbot)

    typer.echo("Chat iniciado. Digite 'sair' para encerrar.")
    while True:
        try:
            text = typer.prompt("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ("sair", "exit", "quit"):
            break

        response = service.chat(text, session_id)
        typer.echo(f"Bot: {response.reply}")
        if response.pending_step:
            typer.echo(f"  [aguardando: {response.pending_step}]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
) -> None:
    """Inicia o servidor HTTP (FastAPI + Uvicorn)."""
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=reload)


def _print_result(result, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({
            "text": result.text,
            "normalised": result.normalised,
            "primary": result.primary,
            "multi_intent": result.multi_intent,
            "confidence": result.confidence,
            "intents": [
                {
                    "intent": m.intent,
                    "label": m.label,
                    "score": m.score,
                    "confidence": m.confidence,
                    "priority": m.priority,
                    "min_score": m.min_score,
                }
                for m in result.intents
            ],
            "language": {
                "language": result.language.language,
                "supported": result.language.supported,
                "reason": result.language.reason,
            },
            "explanation": result.explanation,
        }, ensure_ascii=False))
    else:
        typer.echo(f"primary: {result.primary}")
        typer.echo(f"confidence: {result.confidence:.2%}")
        typer.echo(f"multi_intent: {result.multi_intent}")
        for m in result.intents:
            typer.echo(f"  {m.intent} ({m.label})  score={m.score:.1f}/{m.min_score:.1f}  conf={m.confidence:.2%}")
        typer.echo(f"language: {result.language.language} ({result.language.reason})")
        if result.explanation:
            typer.echo(f"explanation: {result.explanation}")


if __name__ == "__main__":
    app()
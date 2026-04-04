"""
╔══════════════════════════════════════════════════════════════════╗
║              QUICKSY-Go · Sales Assistant Backend                ║
║         Flask + Groq (Llama 3.1 8B) · Production Ready          ║
╚══════════════════════════════════════════════════════════════════╝

Stack   : Python 3.10+ · Flask · Groq · Gunicorn (Render)
Author  : QUICKSY-Go Engineering
Version : 2.0.0
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os
import re
import sys
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from groq import Groq, APIConnectionError, AuthenticationError, RateLimitError


# ─────────────────────────────────────────────
#  LOGGING  (structured, production-grade)
# ─────────────────────────────────────────────
def build_logger() -> logging.Logger:
    """
    Configures a UTF-8 safe, timestamped logger.
    On Render the stream goes straight to the dashboard log viewer.
    """
    log = logging.getLogger("quicksy")
    log.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(fmt)
    log.addHandler(handler)
    log.propagate = False
    return log


logger = build_logger()


# ─────────────────────────────────────────────
#  CONFIGURATION  (single source of truth)
# ─────────────────────────────────────────────
class Config:
    # ── Groq ──────────────────────────────────
    GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL: str          = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_TEMPERATURE: float  = float(os.environ.get("GROQ_TEMPERATURE", "0.72"))
    GROQ_MAX_TOKENS: int     = int(os.environ.get("GROQ_MAX_TOKENS", "400"))

    # ── Context Window Management ─────────────
    # Keep only the last N user/assistant turns to avoid token saturation.
    # Each "turn" = 1 user msg + 1 assistant msg → max 2×N messages kept.
    MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "8"))

    # ── Security ──────────────────────────────
    MAX_MESSAGE_CHARS: int = int(os.environ.get("MAX_MESSAGE_CHARS", "2000"))
    MAX_MESSAGES_PER_REQUEST: int = int(os.environ.get("MAX_MESSAGES_PER_REQUEST", "40"))

    # ── CORS ──────────────────────────────────
    # Comma-separated list of allowed origins, e.g. "https://quicsy-go.com,http://localhost:3000"
    ALLOWED_ORIGINS: list[str] = [
        o.strip()
        for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
        if o.strip()
    ]

    # ── App ───────────────────────────────────
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    PORT: int   = int(os.environ.get("PORT", "5000"))


# ─────────────────────────────────────────────
#  API KEY VALIDATION  (fail-fast, explicit log)
# ─────────────────────────────────────────────
if not Config.GROQ_API_KEY:
    logger.critical(
        "═══════════════════════════════════════════════════════\n"
        "  FATAL — GROQ_API_KEY is not set.\n"
        "  ACTION: Add the secret in Render → Environment → GROQ_API_KEY\n"
        "  The server will start but ALL /chat requests will return 503.\n"
        "═══════════════════════════════════════════════════════"
    )
    # We do NOT sys.exit() here so Render marks the deploy as 'live'
    # and operators can inspect logs; but the endpoint will refuse calls.
    GROQ_CLIENT: Groq | None = None
else:
    try:
        GROQ_CLIENT = Groq(api_key=Config.GROQ_API_KEY)
        logger.info("Groq client initialised OK · model=%s", Config.GROQ_MODEL)
    except Exception as exc:
        logger.critical("Groq client failed to initialise: %s", exc)
        GROQ_CLIENT = None


# ─────────────────────────────────────────────
#  SYSTEM PROMPT  (Harvey Specter · Markdown-free)
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION: str = """
Eres QUICKSY, el estratega principal y primer filtro de la agencia de automatización B2B "QUICKSY-Go".

PERSONALIDAD: Agudo, analítico, seguro de ti mismo y extremadamente directo — estilo Harvey Specter o ejecutivo de Silicon Valley. No suenas a atención al cliente, suenas a un arquitecto de negocios que detecta fugas de dinero e ineficiencias al instante.

REGLAS DE COMPORTAMIENTO:

SITUACIÓN A — El cliente recién cuenta su problema:
1. Empatía de negocios: Valida su dolor con términos de alto nivel (ej. "cuello de botella", "tiempo facturable quemándose", "drenaje de recursos").
2. Pregunta aguda: Haz UNA sola pregunta inteligente que lo haga cuestionar la ineficiencia de su proceso actual.
3. Cierre de apertura: Termina SIEMPRE con: "Podemos entrar más en profundidad y cotizar una solución a tu medida. ¿Quieres que te comunique con Elizabeth vía WhatsApp para revisarlo?"

SITUACIÓN B — El cliente acepta (dice "Sí", "Por favor", "Me interesa", "Claro", "Dale"):
1. Cero rodeos: NO hagas más preguntas. NO repitas el pitch.
2. Confirmación con autoridad: "Excelente decisión de negocios. Toca el botón de WhatsApp que acaba de aparecer aquí abajo y te dejo en manos de Elizabeth para estructurar tu arquitectura."

REGLAS DE FORMATO — CRÍTICAS:
- PROHIBIDO usar markdown: nada de **, ##, *, _, ```, listas con guiones ni numeradas.
- PROHIBIDO usar corchetes, etiquetas HTML, o links de cualquier tipo.
- Escribe ÚNICAMENTE párrafos cortos y fluidos, separados por saltos de línea simples.
- Máximo 3 párrafos por respuesta.
- Nunca uses emojis de estrella, check, flecha ni similares.
""".strip()


# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": Config.ALLOWED_ORIGINS}})

logger.info(
    "Flask app created · debug=%s · allowed_origins=%s",
    Config.DEBUG,
    Config.ALLOWED_ORIGINS,
)


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def sanitize_markdown(text: str) -> str:
    """
    Strip residual Markdown that Llama occasionally emits even when instructed
    not to, so the frontend always receives clean plain text.
    """
    # Bold / italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}",   r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Code blocks / inline code
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`",      r"\1", text)
    # Bullet / numbered lists → keep text, strip marker
    text = re.sub(r"^\s*[-*+]\s+",  "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+",  "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # HTML tags (just in case)
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_history(messages: list[dict], max_turns: int) -> list[dict]:
    """
    Context-window management:
    Keeps only the last `max_turns` user/assistant pairs to avoid
    token overflow while preserving conversational coherence.

    Strategy: always keep the first user message (conversation seed)
    + the last N pairs so QUICKSY remembers what just happened.
    """
    # Separate roles
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= max_turns * 2:
        return non_system  # still within budget, no trimming needed

    # Always preserve the first message (context seed) + recent tail
    first_msg  = non_system[:1]
    tail       = non_system[-(max_turns * 2 - 1):]

    trimmed = first_msg + tail
    logger.debug(
        "History trimmed: original=%d messages → kept=%d (max_turns=%d)",
        len(non_system),
        len(trimmed),
        max_turns,
    )
    return trimmed


def validate_messages(messages: Any) -> tuple[list[dict] | None, str | None]:
    """
    Validates the messages payload.
    Returns (cleaned_messages, None) on success or (None, error_string) on failure.
    """
    if not isinstance(messages, list):
        return None, "El campo 'messages' debe ser una lista."

    if len(messages) == 0:
        return None, "La lista de mensajes está vacía."

    if len(messages) > Config.MAX_MESSAGES_PER_REQUEST:
        return None, (
            f"Demasiados mensajes en la solicitud "
            f"(máximo {Config.MAX_MESSAGES_PER_REQUEST})."
        )

    cleaned: list[dict] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return None, f"El mensaje #{i} no es un objeto válido."

        role    = msg.get("role", "")
        content = msg.get("content", "")

        if role not in {"user", "assistant"}:
            return None, f"El rol '{role}' en el mensaje #{i} no es válido."

        if not isinstance(content, str) or not content.strip():
            return None, f"El contenido del mensaje #{i} está vacío o no es texto."

        if len(content) > Config.MAX_MESSAGE_CHARS:
            return None, (
                f"El mensaje #{i} supera el límite de "
                f"{Config.MAX_MESSAGE_CHARS} caracteres."
            )

        cleaned.append({"role": role, "content": content.strip()})

    return cleaned, None


# ─────────────────────────────────────────────
#  REQUEST TIMING MIDDLEWARE
# ─────────────────────────────────────────────
@app.before_request
def start_timer() -> None:
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    elapsed_ms = (time.perf_counter() - g.start_time) * 1000
    logger.info(
        "%s %s → %d  (%.0f ms)",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ─────────────────────────────────────────────
#  HEALTH CHECK  (Render uptime ping)
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health() -> tuple:
    """
    Lightweight endpoint used by Render's health checks and uptime monitors.
    Returns 200 when the server is alive, 503 if the Groq client is missing.
    """
    status  = "ok" if GROQ_CLIENT else "degraded"
    code    = 200  if GROQ_CLIENT else 503
    payload = {
        "status":    status,
        "service":   "quicksy-go-backend",
        "model":     Config.GROQ_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "groq_ready": GROQ_CLIENT is not None,
    }
    return jsonify(payload), code


# ─────────────────────────────────────────────
#  MAIN CHAT ENDPOINT
# ─────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives a conversation history, prepends the system prompt,
    trims the context window, calls Groq, sanitizes the response,
    and returns clean plain text to the frontend.

    Request body (JSON):
        { "messages": [ {"role": "user"|"assistant", "content": "..."}, ... ] }

    Response (JSON):
        { "reply": "...", "model": "...", "usage": {...} }
    """

    # ── Guard: API key not configured ──────────────────────────
    if not GROQ_CLIENT:
        logger.error(
            "Chat request rejected — GROQ_API_KEY is missing. "
            "Set the environment variable on Render."
        )
        return jsonify({
            "error": "api_key_missing",
            "message": (
                "El servicio no está configurado correctamente. "
                "Contacta al administrador."
            ),
        }), 503

    # ── Parse body ─────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return jsonify({
            "error":   "invalid_json",
            "message": "El cuerpo de la solicitud debe ser JSON válido.",
        }), 400

    # ── Validate messages ──────────────────────────────────────
    raw_messages = body.get("messages")
    messages, validation_error = validate_messages(raw_messages)
    if validation_error:
        logger.warning("Validation error: %s", validation_error)
        return jsonify({
            "error":   "validation_error",
            "message": validation_error,
        }), 422

    # ── Context window trimming ────────────────────────────────
    messages = trim_history(messages, Config.MAX_HISTORY_TURNS)

    # ── Build final message list for Groq ─────────────────────
    groq_messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        *messages,
    ]

    # ── Call Groq ──────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        completion = GROQ_CLIENT.chat.completions.create(
            messages    = groq_messages,
            model       = Config.GROQ_MODEL,
            temperature = Config.GROQ_TEMPERATURE,
            max_tokens  = Config.GROQ_MAX_TOKENS,
        )
        groq_ms = (time.perf_counter() - t0) * 1000

        raw_reply = completion.choices[0].message.content
        reply     = sanitize_markdown(raw_reply)

        usage = {
            "prompt_tokens":     completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens":      completion.usage.total_tokens,
        }

        logger.info(
            "Groq OK · %.0f ms · tokens=%d (prompt=%d + completion=%d)",
            groq_ms,
            usage["total_tokens"],
            usage["prompt_tokens"],
            usage["completion_tokens"],
        )

        return jsonify({
            "reply": reply,
            "model": Config.GROQ_MODEL,
            "usage": usage,
        }), 200

    # ── Specific Groq error types ──────────────────────────────
    except AuthenticationError as exc:
        logger.critical(
            "GROQ AuthenticationError — la API key es inválida o fue revocada. "
            "Actualízala en Render → Environment. Detalle: %s", exc
        )
        return jsonify({
            "error":   "auth_error",
            "message": "Error de autenticación con el proveedor de IA.",
        }), 503

    except RateLimitError as exc:
        logger.warning("GROQ RateLimitError: %s", exc)
        return jsonify({
            "error":   "rate_limit",
            "message": (
                "Estamos recibiendo muchas solicitudes. "
                "Por favor intenta de nuevo en unos segundos."
            ),
        }), 429

    except APIConnectionError as exc:
        logger.error("GROQ APIConnectionError (network issue): %s", exc)
        return jsonify({
            "error":   "connection_error",
            "message": "No se pudo conectar con el servicio de IA. Intenta de nuevo.",
        }), 502

    except Exception as exc:                          # pylint: disable=broad-except
        logger.exception("Unexpected error calling Groq: %s", exc)
        return jsonify({
            "error":   "internal_error",
            "message": (
                "Hubo un error en mis sistemas. "
                "Por favor, contáctanos directamente por WhatsApp."
            ),
        }), 500


# ─────────────────────────────────────────────
#  LEADS ENDPOINT  (future-ready stub)
# ─────────────────────────────────────────────
@app.route("/leads", methods=["POST"])
def collect_lead():
    """
    Stub endpoint for lead capture (Phase 2).
    Validates the payload shape so the frontend can be wired up now
    and the CRM / email integration added later without API changes.

    Request body:
        { "name": "...", "email": "...", "phone": "...", "source": "chat|form" }
    """
    body = request.get_json(silent=True) or {}

    name   = str(body.get("name",   "")).strip()
    email  = str(body.get("email",  "")).strip()
    phone  = str(body.get("phone",  "")).strip()
    source = str(body.get("source", "chat")).strip()

    # Basic validation
    errors = {}
    if not name:
        errors["name"] = "El nombre es obligatorio."
    if not email or "@" not in email:
        errors["email"] = "El email no es válido."

    if errors:
        return jsonify({"error": "validation_error", "fields": errors}), 422

    logger.info(
        "New lead captured · name='%s' · email='%s' · phone='%s' · source='%s'",
        name, email, phone, source,
    )

    # TODO (Phase 2): persist to database / send to HubSpot / trigger email
    return jsonify({
        "status":  "received",
        "message": "Lead registrado correctamente.",
    }), 201


# ─────────────────────────────────────────────
#  NEWSLETTER ENDPOINT  (future-ready stub)
# ─────────────────────────────────────────────
@app.route("/newsletter", methods=["POST"])
def newsletter_subscribe():
    """
    Stub endpoint for newsletter subscription (Phase 2).
    Validates email and logs the subscription.

    Request body:
        { "email": "..." }
    """
    body  = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip()

    if not email or "@" not in email:
        return jsonify({
            "error":   "validation_error",
            "message": "Proporciona un email válido.",
        }), 422

    logger.info("Newsletter subscription · email='%s'", email)

    # TODO (Phase 2): add to Mailchimp / Brevo / custom list
    return jsonify({
        "status":  "subscribed",
        "message": "¡Te has suscrito correctamente!",
    }), 201


# ─────────────────────────────────────────────
#  GLOBAL ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(_err):
    return jsonify({"error": "not_found", "message": "Ruta no encontrada."}), 404


@app.errorhandler(405)
def method_not_allowed(_err):
    return jsonify({"error": "method_not_allowed", "message": "Método HTTP no permitido."}), 405


@app.errorhandler(500)
def internal_error(err):
    logger.exception("Unhandled 500: %s", err)
    return jsonify({"error": "internal_error", "message": "Error interno del servidor."}), 500


# ─────────────────────────────────────────────
#  ENTRYPOINT  (local dev only — Render uses Gunicorn)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(
        "Starting QUICKSY-Go backend · host=0.0.0.0 · port=%d · debug=%s",
        Config.PORT,
        Config.DEBUG,
    )
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
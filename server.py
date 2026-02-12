import os
import secrets
import stripe
import openai
from datetime import datetime, timedelta
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from typing import Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from fastapi import Body
import requests
from contextlib import contextmanager
import json
import stanza
import re
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

def ensure_google_credentials_file():
    """Create a temp service-account JSON file from env var for Google clients."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return

    try:
        json.loads(raw)
    except Exception as e:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON") from e

    path = Path("/tmp/gcp-sa.json")
    path.write_text(raw)
    try:
        path.chmod(0o600)
    except Exception:
        pass
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    print(f"✅ Google Vision credentials loaded from env into {path}")


ensure_google_credentials_file()


# Initialize Stanza pipelines (downloaded on first use)
STANZA_PIPELINES = {}

def get_stanza_pipeline(lang: str):
    """Get or create a Stanza pipeline for the given language."""
    lang_map = {
        "en": "en",
        "grc": "grc",  # Ancient Greek
        "la": "la",    # Latin
    }
    
    stanza_lang = lang_map.get(lang, "en")
    
    if stanza_lang not in STANZA_PIPELINES:
        print(f"📚 Loading Stanza model for {stanza_lang}...")
        stanza.download(stanza_lang, verbose=False)
        STANZA_PIPELINES[stanza_lang] = stanza.Pipeline(
            stanza_lang, 
            processors='tokenize,pos,lemma',
            verbose=False
        )
    
    return STANZA_PIPELINES[stanza_lang]


@contextmanager
def get_db():
    """Get a database cursor with automatic cleanup."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cur
    finally:
        cur.close()
        conn.close()


# Initialize tables on startup
def init_db():
    with get_db() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pro_tokens (
          token TEXT PRIMARY KEY,
          customer_id TEXT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
          id SERIAL PRIMARY KEY,
          customer_id TEXT NOT NULL,
          work_id TEXT NOT NULL,
          section_id TEXT NOT NULL,
          token_id TEXT,
          content TEXT NOT NULL,
          visibility TEXT NOT NULL DEFAULT 'private',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)

        cur.execute("""
        ALTER TABLE annotations
        ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private'
        """)

        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS annotations_unique
        ON annotations (customer_id, work_id, section_id, token_id)
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS restore_tokens (
          token TEXT PRIMARY KEY,
          customer_id TEXT NOT NULL,
          expires_at TIMESTAMPTZ NOT NULL
        )
        """)
        
        # New table for user-uploaded texts
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_texts (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          title TEXT NOT NULL,
          language TEXT NOT NULL,
          sections JSONB NOT NULL,
          created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS user_texts_user_idx ON user_texts (user_id)
        """)

        # Flashcard sets (cloud sync for Pro)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS flashcard_sets (
          user_id TEXT NOT NULL,
          id TEXT NOT NULL,
          name TEXT NOT NULL,
          cards JSONB NOT NULL DEFAULT '[]',
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          created_at TIMESTAMPTZ DEFAULT NOW(),
          PRIMARY KEY (user_id, id)
        )
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS flashcard_sets_user_idx ON flashcard_sets (user_id)
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS podcasts (
          customer_id TEXT NOT NULL,
          work_id TEXT NOT NULL,
          work_title TEXT NOT NULL,
          script TEXT NOT NULL,
          audio BYTEA NOT NULL,
          audio_mime TEXT NOT NULL DEFAULT 'audio/mpeg',
          voice TEXT NOT NULL DEFAULT 'alloy',
          model TEXT NOT NULL DEFAULT 'tts-1',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          PRIMARY KEY (customer_id, work_id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS free_ai_credits (
          user_id TEXT PRIMARY KEY,
          used INT NOT NULL DEFAULT 0,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS email_list (
          email TEXT PRIMARY KEY,
          source TEXT,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)

init_db()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://the-lexicon-project.netlify.app", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
STRIPE_SECRET_KEY = os.environ["STRIPE_SECRET_KEY"]
STRIPE_PRICE_ID = os.environ["STRIPE_PRICE_ID"]
FRONTEND_URL = os.environ["FRONTEND_URL"]
FREE_AI_CREDITS = int(os.environ.get("FREE_AI_CREDITS", "5"))

openai.api_key = OPENAI_API_KEY
stripe.api_key = STRIPE_SECRET_KEY


# -------------------------
# HELPERS
# -------------------------

def customer_from_token(pro_token: str | None):
    if not pro_token:
        return None
    try:
        with get_db() as cur:
            cur.execute(
                "SELECT customer_id FROM pro_tokens WHERE token = %s",
                (pro_token,)
            )
            row = cur.fetchone()
            if row and "customer_id" in row:
                return row["customer_id"]
        return None
    except Exception as e:
        print(f"Error in customer_from_token: {e}")
        return None


def has_pro(pro_token: str | None) -> bool:
    customer_id = customer_from_token(pro_token)
    if not customer_id:
        return False

    subs = stripe.Subscription.list(
        customer=customer_id,
        status="active",
        limit=1
    )

    return len(subs.data) > 0


    return FRONTEND_URL


def get_anon_id(request: Request) -> str | None:
    anon_id = request.headers.get("X-Anon-ID")
    if anon_id and anon_id.startswith("anon_"):
        return anon_id
    return None


def get_free_ai_remaining(request: Request) -> int:
    anon_id = get_anon_id(request)
    if not anon_id:
        raise HTTPException(status_code=401, detail="Anonymous ID required")

    limit = FREE_AI_CREDITS
    with get_db() as cur:
        cur.execute("SELECT used FROM free_ai_credits WHERE user_id = %s", (anon_id,))
        row = cur.fetchone()
        used = row["used"] if row else 0

    return max(limit - used, 0)


def consume_free_ai_credit(request: Request) -> int:
    anon_id = get_anon_id(request)
    if not anon_id:
        raise HTTPException(status_code=401, detail="Anonymous ID required")

    limit = FREE_AI_CREDITS

    with get_db() as cur:
        cur.execute("""
        UPDATE free_ai_credits
        SET used = used + 1, updated_at = NOW()
        WHERE user_id = %s AND used < %s
        RETURNING used
        """, (anon_id, limit))
        row = cur.fetchone()
        if row:
            used = row["used"]
            return max(limit - used, 0)

        cur.execute("""
        INSERT INTO free_ai_credits (user_id, used)
        VALUES (%s, 1)
        ON CONFLICT DO NOTHING
        RETURNING used
        """, (anon_id,))
        row = cur.fetchone()
        if row:
            used = row["used"]
            return max(limit - used, 0)

        cur.execute("SELECT used FROM free_ai_credits WHERE user_id = %s", (anon_id,))
        row = cur.fetchone()
        used = row["used"] if row else limit
        remaining = max(limit - used, 0)

    raise HTTPException(
        status_code=402,
        detail={
            "error": "Free AI credits exhausted",
            "remaining_credits": remaining,
            "limit": limit
        }
    )


def clean_origin(origin: str) -> str:
    """Strip trailing slash and /static suffix from origin."""
    if not origin:
        return ""
    origin = origin.rstrip("/")
    if origin.endswith("/static"):
        origin = origin[:-7]
    return origin.rstrip("/")


def get_frontend_origin(request: Request) -> str:
    """Best-effort origin for links (use request origin, then referer, else default)."""
    origin = request.headers.get("origin")
    if origin:
        return clean_origin(origin)

    referer = request.headers.get("referer")
    if referer:
        parsed = urlparse(referer)
        if parsed.scheme and parsed.netloc:
            return clean_origin(f"{parsed.scheme}://{parsed.netloc}")

    return clean_origin(FRONTEND_URL)


# -------------------------
# MODELS
# -------------------------
class ExplainWord(BaseModel):
    token: str
    lemma: str
    pos: str
    morph: str
    sentence: str
    speaker: str | None = ""
    work: str | None = ""
    prompt_memory: Optional[str] = ""


class ExplainPassage(BaseModel):
    work: str
    section: str
    speaker: Optional[str] = ""
    greek: str
    translation: Optional[str] = ""
    prompt_memory: Optional[str] = ""


class LinkWord(BaseModel):
    work_id: Optional[str] = ""
    work_title: Optional[str] = ""
    section_label: Optional[str] = ""
    token: str
    lemma: Optional[str] = ""
    pos: Optional[str] = ""
    morph: Optional[str] = ""
    sentence: Optional[str] = ""


class ExplainLink(BaseModel):
    from_word: LinkWord = Field(..., alias="from")
    to_word: LinkWord = Field(..., alias="to")
    link_note: Optional[str] = ""
    prompt_memory: Optional[str] = ""


class SaveAnnotation(BaseModel):
    work_id: str
    section_id: str
    token_id: Optional[str] = None
    content: str
    visibility: Optional[str] = "private"


class Flashcard(BaseModel):
    id: str
    work_id: Optional[str] = None
    section_id: Optional[str] = None
    token_id: Optional[str] = None
    text: Optional[str] = ""
    lemma: Optional[str] = ""
    gloss: Optional[str] = ""
    morph: Optional[str] = ""
    context: Optional[str] = ""
    translation: Optional[str] = ""
    created_at: Optional[str] = ""


class FlashcardSet(BaseModel):
    id: str
    name: str
    cards: List[Flashcard] = []
    updated_at: Optional[str] = None


class FlashcardSyncRequest(BaseModel):
    sets: List[FlashcardSet] = []


class PodcastGenerateRequest(BaseModel):
    work_id: str
    title: str
    author: Optional[str] = ""
    meta: Optional[str] = ""
    sections: List[dict] = []
    target_minutes: Optional[int] = 10
    voices: Optional[List[str]] = ["alloy", "nova"]


class AnnotationRow(BaseModel):
    section: Optional[str] = ""
    token: Optional[str] = ""
    lemma: Optional[str] = ""
    body: Optional[str] = ""


class EmailAnnotationsRequest(BaseModel):
    email: str
    work_id: Optional[str] = ""
    work_title: Optional[str] = ""
    rows: List[AnnotationRow] = []


# -------------------------
# PODCAST HELPERS
# -------------------------

MAX_PODCAST_SOURCE_CHARS = 30000


def build_podcast_source_text(sections: List[dict]) -> str:
    """Build a capped source text from section payloads."""
    total = 0
    parts = []
    for sec in sections or []:
        label = sec.get("label") or sec.get("id") or ""
        text = sec.get("text") or ""
        if not text:
            continue
        block = f"[{label}]\n{text}\n\n"
        if total + len(block) > MAX_PODCAST_SOURCE_CHARS:
            remaining = MAX_PODCAST_SOURCE_CHARS - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def generate_podcast_script(req: PodcastGenerateRequest, source_text: str) -> str:
    """Generate a two-host podcast script from source text using OpenAI."""
    prompt = f"""
You are creating a polished, listenable podcast episode for students of classical texts.
Write a dialogue between two hosts. Use explicit speaker tags at the start of each line,
exactly as "HOST A:" or "HOST B:" (all caps). No bullet points.
Aim for ~{req.target_minutes} minutes (roughly 1200-1600 words).
Focus on: context, key themes, character arcs, and 2–3 close-reading moments.
Avoid quoting long passages; keep quotes very short.
If the source is partial, be transparent and avoid over-claiming.

Work: {req.title}
Author: {req.author}
Meta: {req.meta}

Source excerpt:
{source_text}
"""

    r = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classical philologist and podcast writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1800,
    )

    return r.choices[0].message.content.strip()


def split_text_for_tts(text: str, max_chars: int = 3800) -> List[str]:
    """Split text into chunks within TTS limits, preferring paragraph boundaries."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # Fallback to sentence-ish splits for very long paragraphs
            sentences = re.split(r'(?<=[.!?])\s+', para)
        else:
            sentences = [para]

        for sent in sentences:
            if not sent:
                continue
            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if len(sent) <= max_chars:
                current = sent
            else:
                # Hard split
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
                current = ""

    if current:
        chunks.append(current)
    return chunks


def split_dialogue(script: str) -> List[dict]:
    """Split script into speaker-tagged segments."""
    segments = []
    current_speaker = None
    current_text = ""
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("HOST A:") or line.startswith("HOST B:"):
            if current_text and current_speaker:
                segments.append({"speaker": current_speaker, "text": current_text.strip()})
            if line.startswith("HOST A:"):
                current_speaker = "A"
                current_text = line.replace("HOST A:", "", 1).strip()
            else:
                current_speaker = "B"
                current_text = line.replace("HOST B:", "", 1).strip()
        else:
            current_text = f"{current_text} {line}".strip() if current_text else line
    if current_text and current_speaker:
        segments.append({"speaker": current_speaker, "text": current_text.strip()})
    return segments


def generate_podcast_audio(script: str, voice: str = "alloy", model: str = "tts-1", voices: Optional[List[str]] = None):
    """Generate TTS audio bytes for a script, alternating voices by speaker."""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    voice_list = voices or [voice, "nova"]
    voice_a = voice_list[0] if len(voice_list) > 0 else voice
    voice_b = voice_list[1] if len(voice_list) > 1 else voice

    segments = split_dialogue(script)
    if not segments:
        segments = [{"speaker": "A", "text": script}]

    audio_parts = []
    part_idx = 0
    for seg in segments:
        chosen_voice = voice_a if seg["speaker"] == "A" else voice_b
        chunks = split_text_for_tts(seg["text"], max_chars=3800)
        for chunk in chunks:
            part_idx += 1
            payload = {
                "model": model,
                "voice": chosen_voice,
                "input": chunk,
                "response_format": "mp3"
            }
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if not res.ok:
                raise HTTPException(status_code=500, detail=f"TTS failed (part {part_idx}): {res.text}")
            audio_parts.append(res.content)

    return b"".join(audio_parts), "audio/mpeg", ",".join([voice_a, voice_b]), model


# -------------------------
# EMAIL
# -------------------------

def send_restore_email(to_email: str, restore_url: str):
    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "from": "Lexikon <hello@the-lexicon-project.com>",
                "to": to_email,
                "subject": "Restore your Lexikon subscription",
                "html": f"""
                    <p>You can restore your Lexikon Pro access by clicking the link below:</p>
                    <p><a href="{restore_url}">Restore my subscription</a></p>
                    <p>This link can be used once and will expire.</p>
                """
            },
            timeout=5,
        )
        
        if response.ok:
            print(f"✅ Restore email sent to {to_email}")
        else:
            print(f"❌ Resend API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Failed to send restore email: {e}")


def html_escape(val: str) -> str:
    return (
        str(val or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def send_annotations_email(to_email: str, work_title: str, rows: List[AnnotationRow]):
    safe_title = html_escape(work_title or "Annotations")
    row_html = "".join(
        f"<tr>"
        f"<td>{html_escape(r.section)}</td>"
        f"<td>{html_escape(r.token)}</td>"
        f"<td>{html_escape(r.lemma)}</td>"
        f"<td>{html_escape(r.body)}</td>"
        f"</tr>"
        for r in rows
    )
    html = f"""
        <p>Here are your Lexikon annotations for <strong>{safe_title}</strong>.</p>
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr>
              <th style="text-align:left;border-bottom:1px solid #ddd;padding:6px;">Section</th>
              <th style="text-align:left;border-bottom:1px solid #ddd;padding:6px;">Token</th>
              <th style="text-align:left;border-bottom:1px solid #ddd;padding:6px;">Lemma</th>
              <th style="text-align:left;border-bottom:1px solid #ddd;padding:6px;">Note</th>
            </tr>
          </thead>
          <tbody>
            {row_html}
          </tbody>
        </table>
        <p style="color:#666;font-size:12px;">Exported from Lexikon.</p>
    """

    text_lines = ["Lexikon annotations", f"Work: {work_title}", ""]
    for r in rows:
        text_lines.append(f"{r.section} | {r.token} | {r.lemma} | {r.body}")
    text = "\n".join(text_lines)

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "from": "Lexikon <hello@the-lexicon-project.com>",
                "to": to_email,
                "subject": f"Your Lexikon annotations — {work_title}",
                "html": html,
                "text": text,
            },
            timeout=8,
        )

        if response.ok:
            print(f"✅ Annotation email sent to {to_email}")
        else:
            print(f"❌ Resend API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=502, detail="Email failed")

    except Exception as e:
        print(f"❌ Failed to send annotations email: {e}")
        raise HTTPException(status_code=502, detail="Email failed")


@app.post("/email/annotations")
def email_annotations(payload: EmailAnnotationsRequest):
    email = (payload.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No annotations to send")

    with get_db() as cur:
        cur.execute(
            """
            INSERT INTO email_list (email, source, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (email) DO UPDATE
            SET source = EXCLUDED.source,
                updated_at = NOW()
            """,
            (email, f"export:{payload.work_id or 'unknown'}"),
        )

    send_annotations_email(email, payload.work_title or "Annotations", payload.rows)
    return {"ok": True}


# -------------------------
# STRIPE ENDPOINTS
# -------------------------

@app.get("/create-checkout-session")
def create_checkout_session(request: Request):
    origin = request.headers.get("origin") or "https://the-lexicon-project.netlify.app"

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        billing_address_collection="required",
        success_url=f"{origin}/app.html?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{origin}/index.html",
    )

    return {"url": session.url}


STRIPE_WEBHOOK_SECRET = os.environ["STRIPE_WEBHOOK_SECRET"]

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session["customer"]
        email = session["customer_details"]["email"]
        origin = get_frontend_origin(request)

        restore_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)

        with get_db() as cur:
            cur.execute("""
                INSERT INTO restore_tokens (token, customer_id, expires_at)
                VALUES (%s, %s, %s)
            """, (restore_token, customer_id, expires_at))

        restore_url = f"{origin}/app.html?restore_token={restore_token}"
        send_restore_email(email, restore_url)

    return {"ok": True}


@app.get("/checkout-success")
def checkout_success(session_id: str, request: Request):
    try:
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=["customer"]
        )

        customer_id = session.customer.id
        email = (
            session.customer_details.email
            if session.customer_details and session.customer_details.email
            else None
        )

        if not customer_id:
            raise HTTPException(status_code=400, detail="Missing customer")

        pro_token = secrets.token_urlsafe(32)
        restore_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)

        with get_db() as cur:
            cur.execute(
                "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
                (pro_token, customer_id)
            )
            cur.execute("""
                INSERT INTO restore_tokens (token, customer_id, expires_at)
                VALUES (%s, %s, %s)
            """, (restore_token, customer_id, expires_at))

        origin = get_frontend_origin(request)
        restore_url = f"{origin}/app.html?restore_token={restore_token}"

        print("RESTORE LINK:", restore_url)
        print("CHECKOUT EMAIL:", email)

        # if email:
        #     print(f"📧 Attempting to send restore email to {email}...")
        #     send_restore_email(email, restore_url)
        # else:
        #     print("⚠️ No email found, skipping restore email")

        return {"pro_token": pro_token}

    except HTTPException as e:
        raise e
    except Exception as e:
        print("CHECKOUT ERROR:", e)
        return {"error": str(e)}


# -------------------------
# BILLING ENDPOINTS
# -------------------------

@app.post("/billing/request-restore")
async def request_restore(request: Request):
    payload = await request.json()
    email = payload.get("email")

    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    customers = stripe.Customer.search(
        query=f"email:'{email}'",
        limit=1
    ).data

    if not customers:
        print(f"⚠️ No customer found for {email}")
        return {"ok": True}

    customer = customers[0]

    subs = stripe.Subscription.list(
        customer=customer.id,
        status="active",
        limit=1
    ).data

    if not subs:
        print(f"⚠️ No active subscription for {email}")
        return {"ok": True}

    restore_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)

    with get_db() as cur:
        cur.execute("""
            INSERT INTO restore_tokens (token, customer_id, expires_at)
            VALUES (%s, %s, %s)
        """, (restore_token, customer.id, expires_at))

    origin = get_frontend_origin(request)
    restore_url = f"{origin}/app.html?restore_token={restore_token}"

    print(f"📧 Sending restore email to {email}...")
    send_restore_email(email, restore_url)

    return {"ok": True}


@app.post("/billing/restore-from-link")
async def restore_from_link(request: Request):
    payload = await request.json()
    token = payload.get("restore_token")

    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

    with get_db() as cur:
        cur.execute("""
            SELECT customer_id
            FROM restore_tokens
            WHERE token = %s
              AND expires_at > NOW()
        """, (token,))

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Invalid or expired link")

        customer_id = row["customer_id"]

        subs = stripe.Subscription.list(
            customer=customer_id,
            status="active",
            limit=1
        ).data

        if not subs:
            raise HTTPException(status_code=402, detail="No active subscription")

        pro_token = secrets.token_urlsafe(32)

        cur.execute(
            "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
            (pro_token, customer_id)
        )

        cur.execute("DELETE FROM restore_tokens WHERE token = %s", (token,))

    return {"pro_token": pro_token}


@app.get("/billing/portal")
def billing_portal(request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401)

    origin = request.headers.get("origin") or "https://the-lexicon-project.netlify.app"

    portal = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=f"{origin}/app.html"
    )

    return {"url": portal.url}


@app.get("/billing/restore-token")
def billing_restore_token(session_id: str):
    session = stripe.checkout.Session.retrieve(session_id)

    customer_id = session.customer
    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer on session")

    subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
    if not subs.data:
        raise HTTPException(status_code=402, detail="No active subscription")

    token = secrets.token_urlsafe(32)

    with get_db() as cur:
        cur.execute(
            "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
            (token, customer_id)
        )

    return {"pro_token": token}


# -------------------------
# AI ENDPOINTS
# -------------------------

@app.get("/ai/credits")
def ai_credits(request: Request):
    pro = request.headers.get("X-Pro-Token")
    if has_pro(pro):
        return {"pro": True, "remaining_credits": None, "limit": None}

    remaining = get_free_ai_remaining(request)
    return {"pro": False, "remaining_credits": remaining, "limit": FREE_AI_CREDITS}


@app.post("/ai/explain-word")
def explain_word(req: ExplainWord, request: Request):
    try:
        pro = request.headers.get("X-Pro-Token")
        remaining = None
        if not has_pro(pro):
            remaining = consume_free_ai_credit(request)
        memory = (req.prompt_memory or "").strip()
        memory_block = ""
        if memory:
            memory_block = f"""
Student saved annotations (prompt memory):
{memory}
Use this as supporting context when helpful, but prioritize the source text and morphology.
"""

        prompt = f"""
You are a classical languages tutor. Give a concise, pointed coaching note (not an info dump).
Goal: help a student see the key grammatical cue and how it fits the sentence; then suggest a quick follow-up question.
Be brief (4-6 sentences max).
Include:
- What the form tells us (lemma, POS, morphology).
- One or two likely syntactic roles.
- One micro-question to check understanding.
Word: {req.token}
Lemma: {req.lemma}
POS: {req.pos}
Morphology: {req.morph}
Context word/phrase: {req.sentence}
{memory_block}
"""

        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a classical philologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )

        return {
            "explanation": r.choices[0].message.content.strip(),
            "remaining_credits": remaining,
            "limit": FREE_AI_CREDITS
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("AI ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/ai/explain-link")
def explain_link(req: ExplainLink, request: Request):
    try:
        pro = request.headers.get("X-Pro-Token")
        remaining = None
        if not has_pro(pro):
            remaining = consume_free_ai_credit(request)

        a = req.from_word
        b = req.to_word
        memory = (req.prompt_memory or "").strip()
        link_note = (req.link_note or "").strip()
        memory_block = ""
        if memory:
            memory_block = f"""
Student saved annotations (prompt memory):
{memory}
Use this as supporting context when helpful, but prioritize the source text and morphology.
"""
        link_note_block = ""
        if link_note:
            link_note_block = f"""
Student link note:
{link_note}
"""

        prompt = f"""
You are a classical languages tutor. The student linked two words and wants a brief explanation of the link.
Be concise (4-6 sentences). Focus on:
- What each form tells us (lemma, POS, morphology).
- How the two words relate (syntactic, semantic, thematic, or rhetorical).
- One quick check-for-understanding question.

Word A: {a.token}
Lemma A: {a.lemma}
POS A: {a.pos}
Morph A: {a.morph}
Context A: {a.sentence}
Work A: {a.work_title} {a.section_label}

Word B: {b.token}
Lemma B: {b.lemma}
POS B: {b.pos}
Morph B: {b.morph}
Context B: {b.sentence}
Work B: {b.work_title} {b.section_label}
{link_note_block}
{memory_block}
"""

        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a classical philologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=320,
        )

        return {
            "explanation": r.choices[0].message.content.strip(),
            "remaining_credits": remaining,
            "limit": FREE_AI_CREDITS
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("AI ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/ai/explain-passage")
def explain_passage(req: ExplainPassage, request: Request):
    try:
        pro = request.headers.get("X-Pro-Token")
        remaining = None
        if not has_pro(pro):
            remaining = consume_free_ai_credit(request)
        memory = (req.prompt_memory or "").strip()
        memory_block = ""
        if memory:
            memory_block = f"""
Student saved annotations (prompt memory):
{memory}
Use this as supporting context when helpful, but prioritize the source text and syntax.
"""

        prompt = f"""
You are a classical languages tutor. Give a short, structured walkthrough to help a student read the passage (not a summary).
Keep it to 5-8 sentences. Focus on:
- Sentence spine: finite verbs and clauses (who does what).
- Two or three tricky constructions or particles to watch.
- One quick check-for-understanding question at the end.
Avoid paraphrase; stay on syntax and how to navigate it.

Work: {req.work}
Section: {req.section}
Speaker: {req.speaker}

Greek text:
{req.greek}

Translation (for reference only):
{req.translation}
{memory_block}
"""

        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a classical philologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )

        return {
            "explanation": r.choices[0].message.content.strip(),
            "remaining_credits": remaining,
            "limit": FREE_AI_CREDITS
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("AI ERROR (passage):", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# -------------------------
# PODCAST ENDPOINTS
# -------------------------

@app.get("/podcasts/{work_id}")
def get_podcast_meta(work_id: str, request: Request):
    customer_id = require_pro_user(request)
    with get_db() as cur:
        cur.execute("""
        SELECT work_id, created_at, updated_at, audio_mime
        FROM podcasts
        WHERE customer_id = %s AND work_id = %s
        """, (customer_id, work_id))
        row = cur.fetchone()

    if not row:
        return {"exists": False}

    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/podcasts/{work_id}/audio"
    return {
        "exists": True,
        "work_id": work_id,
        "audio_url": audio_url,
        "audio_mime": row.get("audio_mime"),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None
    }


@app.get("/podcasts/{work_id}/audio")
def get_podcast_audio(work_id: str, request: Request):
    # Audio tag can't send headers, so allow token via query param
    token = request.headers.get("X-Pro-Token") or request.query_params.get("token")
    customer_id = customer_from_token(token)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Pro token required")
    if not has_pro(token):
        raise HTTPException(status_code=402, detail="Pro subscription required")
    with get_db() as cur:
        cur.execute("""
        SELECT audio, audio_mime
        FROM podcasts
        WHERE customer_id = %s AND work_id = %s
        """, (customer_id, work_id))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Podcast not found")

    audio_data = row["audio"]
    if isinstance(audio_data, memoryview):
        audio_data = audio_data.tobytes()
    size = len(audio_data) if audio_data else 0
    print(f"🎧 audio bytes={size} work_id={work_id} customer_id={customer_id}")

    return Response(
        content=audio_data,
        media_type=row.get("audio_mime") or "audio/mpeg",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Length": str(size)
        }
    )


@app.post("/podcasts/generate")
def generate_podcast(req: PodcastGenerateRequest, request: Request):
    try:
        print("🎙️ Podcast generate: start")
        customer_id = require_pro_user(request)
        print(f"🎙️ customer_id={customer_id} work_id={req.work_id}")

        if not req.sections:
            print("⚠️ No sections provided")
            raise HTTPException(status_code=400, detail="No sections provided")

        source_text = build_podcast_source_text(req.sections)
        print(f"🎙️ source chars={len(source_text)} sections={len(req.sections)}")
        if not source_text:
            print("⚠️ No source text available")
            raise HTTPException(status_code=400, detail="No source text available")

        print("🎙️ generating script…")
        script = generate_podcast_script(req, source_text)
        print(f"🎙️ script length={len(script)}")

        print("🎙️ generating audio…")
        audio_bytes, audio_mime, voice, model = generate_podcast_audio(
            script,
            voices=req.voices or ["alloy", "nova"]
        )
        print(f"🎙️ audio bytes={len(audio_bytes)} mime={audio_mime} voice={voice} model={model}")

        with get_db() as cur:
            print("🎙️ writing to DB…")
            cur.execute("""
            INSERT INTO podcasts (customer_id, work_id, work_title, script, audio, audio_mime, voice, model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (customer_id, work_id) DO UPDATE SET
              work_title = EXCLUDED.work_title,
              script = EXCLUDED.script,
              audio = EXCLUDED.audio,
              audio_mime = EXCLUDED.audio_mime,
              voice = EXCLUDED.voice,
              model = EXCLUDED.model,
              updated_at = NOW()
            """, (customer_id, req.work_id, req.title, script, psycopg2.Binary(audio_bytes), audio_mime, voice, model))

        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/podcasts/{req.work_id}/audio"
        print(f"🎙️ done audio_url={audio_url}")
        return {
            "ok": True,
            "work_id": req.work_id,
            "audio_url": audio_url,
            "updated_at": datetime.utcnow().isoformat()
        }
    except HTTPException as e:
        print(f"❌ Podcast generate HTTP error: {e.detail}")
        raise e
    except Exception as e:
        print(f"❌ Podcast generate error: {e}")
        raise e


# -------------------------
# ANNOTATION ENDPOINTS
# -------------------------

@app.post("/annotations")
def save_annotation(req: SaveAnnotation, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Only Pro users can make annotations public
    visibility = req.visibility or "private"
    if visibility == "public" and not has_pro(pro):
        visibility = "private"

    with get_db() as cur:
        if not req.content.strip():
            cur.execute("""
            DELETE FROM annotations
            WHERE customer_id = %s
              AND work_id = %s
              AND section_id = %s
              AND token_id IS NOT DISTINCT FROM %s
            """, (customer_id, req.work_id, req.section_id, req.token_id))
            return {"deleted": True}

        cur.execute("""
        INSERT INTO annotations (customer_id, work_id, section_id, token_id, content, visibility)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id, work_id, section_id, token_id)
        DO UPDATE SET content = EXCLUDED.content, visibility = EXCLUDED.visibility, updated_at = NOW()
        """, (customer_id, req.work_id, req.section_id, req.token_id, req.content, visibility))

    return {"ok": True}


@app.get("/annotations")
def get_annotations(work_id: str, section_id: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)

    if not customer_id:
        return {"annotations": {}}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT token_id, content
            FROM annotations
            WHERE customer_id = %s
              AND work_id = %s
              AND section_id = %s
            """, (customer_id, work_id, section_id))

            rows = cur.fetchall()

            out = {}
            for r in rows:
                if r and "token_id" in r and "content" in r:
                    out[r["token_id"]] = r["content"]

            return {"annotations": out}
    except Exception as e:
        print(f"Error fetching annotations: {e}")
        return {"annotations": {}}


@app.get("/annotations/work")
def get_work_annotations(work_id: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)

    if not customer_id:
        return {"annotations": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT section_id, token_id, content
            FROM annotations
            WHERE customer_id = %s
              AND work_id = %s
            """, (customer_id, work_id))

            rows = cur.fetchall()
            out = []
            for r in rows:
                if not r:
                    continue
                out.append({
                    "section_id": r.get("section_id"),
                    "token_id": r.get("token_id"),
                    "content": r.get("content")
                })
            return {"annotations": out}
    except Exception as e:
        print(f"Error fetching work annotations: {e}")
        return {"annotations": []}


@app.get("/annotations/all")
def get_all_annotations(request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)

    if not customer_id:
        return {"annotations": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT work_id, section_id, token_id, content
            FROM annotations
            WHERE customer_id = %s
            """, (customer_id,))

            rows = cur.fetchall()
            out = []
            for r in rows:
                if not r:
                    continue
                out.append({
                    "work_id": r.get("work_id"),
                    "section_id": r.get("section_id"),
                    "token_id": r.get("token_id"),
                    "content": r.get("content")
                })
            return {"annotations": out}
    except Exception as e:
        print(f"Error fetching all annotations: {e}")
        return {"annotations": []}


@app.get("/annotations/public")
def get_public_annotations(work_id: str, section_id: str):
    try:
        with get_db() as cur:
            cur.execute("""
            SELECT token_id, content
            FROM annotations
            WHERE work_id = %s
              AND section_id = %s
              AND visibility = 'public'
            """, (work_id, section_id))

            rows = cur.fetchall()

            out = {}
            for r in rows:
                if r and "token_id" in r and "content" in r:
                    out[r["token_id"]] = r["content"]

            return {"annotations": out}
    except Exception as e:
        print(f"Error fetching public annotations: {e}")
        return {"annotations": {}}


@app.get("/annotations/search")
def search_annotations(q: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"results": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT work_id, section_id, token_id, content
            FROM annotations
            WHERE customer_id = %s
              AND content ILIKE %s
            ORDER BY updated_at DESC
            LIMIT 50
            """, (customer_id, f"%{q}%"))

            rows = cur.fetchall()
            return {"results": rows if rows else []}
    except Exception as e:
        print(f"Error searching annotations: {e}")
        return {"results": []}


@app.get("/restore-pro")
def restore_pro(session_id: str):
    session = stripe.checkout.Session.retrieve(session_id)
    customer_id = session.customer

    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer")

    subs = stripe.Subscription.list(
        customer=customer_id,
        status="active",
        limit=1
    )

    if not subs.data:
        raise HTTPException(status_code=402, detail="No active subscription")

    token = secrets.token_urlsafe(32)

    with get_db() as cur:
        cur.execute("""
        INSERT INTO pro_tokens (token, customer_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """, (token, customer_id))

    return {"pro_token": token}


# -------------------------
# USER TEXT IMPORT ENDPOINTS
# -------------------------

def get_user_id(request: Request) -> str | None:
    """Get user ID from pro token, or from X-Anon-ID header for non-pro users."""
    # First try pro token
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if customer_id:
        return customer_id
    
    # Fall back to anonymous ID from header (generated and stored client-side)
    anon_id = request.headers.get("X-Anon-ID")
    if anon_id and anon_id.startswith("anon_"):
        return anon_id
    
    return None


def require_pro_user(request: Request, allow_inactive: bool = False) -> str:
    """Return customer id from pro token. Optionally skip active sub check."""
    token = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(token)
    if not customer_id:
        raise HTTPException(status_code=402, detail="Pro required")

    if allow_inactive:
        return customer_id

    if not has_pro(token):
        raise HTTPException(status_code=402, detail="No active subscription")

    return customer_id


def count_words(text: str) -> int:
    return len(text.split())


def process_text_with_stanza(text: str, language: str) -> List[dict]:
    """Process text with Stanza and return sections with tokens."""
    raw_sections = re.split(r'\n\s*\n', text.strip())

    if len(raw_sections) <= 1:
        words = text.split()
        raw_sections = []
        for i in range(0, len(words), 500):
            raw_sections.append(' '.join(words[i:i + 500]))

    raw_sections = raw_sections[:50]

    nlp = get_stanza_pipeline(language)
    sections = []

    for idx, section_text in enumerate(raw_sections):
        section_text = section_text.strip()
        if not section_text:
            continue

        doc = nlp(section_text)
        tokens = []
        token_idx = 0

        for sentence in doc.sentences:
            for word in sentence.words:
                tokens.append({
                    "id": f"w{token_idx}",
                    "t": word.text,
                    "lemma": word.lemma or word.text,
                    "pos": word.upos or "",
                    "morph": word.feats or ""
                })
                token_idx += 1

        sections.append({
            "id": f"section_{idx + 1}",
            "label": f"Section {idx + 1}",
            "tokens": tokens,
            "translation": ""
        })

    return sections


def _get_vision_client():
    try:
        from google.cloud import vision  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail="google-cloud-vision not installed") from e
    return vision.ImageAnnotatorClient()


@app.post("/ocr/vision")
async def ocr_vision(
    request: Request,
    file: UploadFile = File(...)
):
    """Run Google Vision OCR (DOCUMENT_TEXT_DETECTION) on an image."""
    if not file:
        raise HTTPException(status_code=400, detail="No image provided")

    # Basic content-type check
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")

    # 5MB limit to avoid surprise costs and timeouts
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 5MB)")

    client = _get_vision_client()

    try:
        from google.cloud import vision  # type: ignore
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
    except Exception as e:
        print(f"❌ Vision OCR request failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Vision OCR request failed: {e}") from e

    if response.error and response.error.message:
        raise HTTPException(status_code=500, detail=f"Vision OCR error: {response.error.message}")

    text = ""
    if response.full_text_annotation and response.full_text_annotation.text:
        text = response.full_text_annotation.text

    return {"text": text}


@app.post("/texts/upload")
async def upload_text(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    language: str = Form(...)
):
    """Upload and process a text file. Processing is free for all users."""
    user_id = get_user_id(request)

    try:
        content = await file.read()
        text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    word_count = count_words(text)
    if word_count > 10000:
        raise HTTPException(status_code=400, detail=f"Text too long ({word_count} words). Maximum is 10,000 words.")
    if word_count < 10:
        raise HTTPException(status_code=400, detail="Text too short. Please upload at least 10 words.")

    if language not in ["en", "grc", "la"]:
        raise HTTPException(status_code=400, detail="Language must be 'en', 'grc', or 'la'")

    try:
        user_display = (user_id[:20] + "...") if user_id else "anonymous"
        print(f"📝 Processing {word_count} words in {language} for user {user_display}...")
        sections = process_text_with_stanza(text, language)
        print(f"✅ Created {len(sections)} sections")
    except Exception as e:
        print(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    text_id = f"user_{secrets.token_urlsafe(12)}"

    if user_id:
        with get_db() as cur:
            cur.execute("""
            INSERT INTO user_texts (id, user_id, title, language, sections)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                sections = EXCLUDED.sections
            """, (text_id, user_id, title, language, Json(sections)))

    return {
        "id": text_id,
        "title": title,
        "language": language,
        "sections": sections,
        "section_count": len(sections),
        "word_count": word_count
    }


class SaveTextRequest(BaseModel):
    id: str
    title: str
    language: str
    sections: List[dict]


@app.post("/texts/save")
def save_text(req: SaveTextRequest, request: Request):
    """Save a pre-processed text to cloud (for migration/sync). Pro only."""
    customer_id = require_pro_user(request)

    with get_db() as cur:
        cur.execute("""
        INSERT INTO user_texts (id, user_id, title, language, sections)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            sections = EXCLUDED.sections
        """, (req.id, customer_id, req.title, req.language, Json(req.sections)))

    return {"ok": True}


@app.get("/texts")
def list_user_texts(request: Request):
    """List all texts uploaded by the current user."""
    user_id = get_user_id(request)
    if not user_id:
        return {"texts": []}

    with get_db() as cur:
        cur.execute("""
        SELECT id, title, language, created_at,
               jsonb_array_length(sections) as section_count
        FROM user_texts
        WHERE user_id = %s
        ORDER BY created_at DESC
        """, (user_id,))

        rows = cur.fetchall()
        texts = []
        for r in rows or []:
            texts.append({
                "id": r["id"],
                "title": r["title"],
                "language": r["language"],
                "section_count": r["section_count"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None
            })

        return {"texts": texts}


@app.get("/texts/{text_id}")
def get_user_text(text_id: str, request: Request):
    """Get a specific user text with all sections."""
    user_id = get_user_id(request)

    with get_db() as cur:
        cur.execute("""
        SELECT id, user_id, title, language, sections
        FROM user_texts
        WHERE id = %s
        """, (text_id,))

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Text not found")

        if user_id and row["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            "id": row["id"],
            "title": row["title"],
            "language": row["language"],
            "sections": row["sections"]
        }


@app.delete("/texts/{text_id}")
def delete_user_text(text_id: str, request: Request):
    """Delete a user text."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    with get_db() as cur:
        cur.execute("""
        SELECT user_id FROM user_texts WHERE id = %s
        """, (text_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Text not found")
        if row["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        cur.execute("DELETE FROM user_texts WHERE id = %s", (text_id,))
        cur.execute("""
        DELETE FROM annotations WHERE work_id = %s AND customer_id = %s
        """, (text_id, user_id))

    return {"ok": True}


@app.get("/flashcards")
def list_flashcards(request: Request):
    """Return all flashcard sets for the current Pro user."""
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"sets": []}

    with get_db() as cur:
        cur.execute("""
        SELECT id, name, cards, updated_at
        FROM flashcard_sets
        WHERE user_id = %s
        ORDER BY updated_at DESC
        """, (customer_id,))
        rows = cur.fetchall() or []
        return {"sets": rows}


@app.post("/flashcards/sync")
def sync_flashcards(req: FlashcardSyncRequest, request: Request):
    """Replace all flashcard sets for the user (used for migration/sync)."""
    customer_id = require_pro_user(request)

    with get_db() as cur:
        cur.execute("""
        DELETE FROM flashcard_sets WHERE user_id = %s
        """, (customer_id,))

        for s in req.sets or []:
            updated_at = s.updated_at or datetime.utcnow().isoformat()
            cur.execute("""
            INSERT INTO flashcard_sets (user_id, id, name, cards, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, id)
            DO UPDATE SET
              name = EXCLUDED.name,
              cards = EXCLUDED.cards,
              updated_at = EXCLUDED.updated_at
            """, (
                customer_id,
                s.id,
                s.name,
                Json([c.dict() if hasattr(c, "dict") else c for c in (s.cards or [])]),
                updated_at
            ))

    return {"ok": True}


@app.delete("/flashcards")
def delete_all_flashcards(request: Request):
    """Delete all flashcard sets for a user (used when moving off Pro)."""
    customer_id = require_pro_user(request, allow_inactive=True)

    with get_db() as cur:
        cur.execute("""
        DELETE FROM flashcard_sets
        WHERE user_id = %s
        """, (customer_id,))

    return {"ok": True}

import os
import secrets
import stripe
import openai
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
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
        CREATE TABLE IF NOT EXISTS phrase_annotations (
          customer_id TEXT NOT NULL,
          id TEXT NOT NULL,
          work_id TEXT NOT NULL,
          section_id TEXT NOT NULL,
          token_ids JSONB NOT NULL DEFAULT '[]',
          phrase_text TEXT NOT NULL DEFAULT '',
          meaning_text TEXT NOT NULL DEFAULT '',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          PRIMARY KEY (customer_id, id)
        )
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS phrase_annotations_customer_idx
        ON phrase_annotations (customer_id, updated_at DESC)
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS phrase_annotations_work_section_idx
        ON phrase_annotations (customer_id, work_id, section_id)
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS text_connections (
          customer_id TEXT NOT NULL,
          id TEXT NOT NULL,
          from_word JSONB NOT NULL,
          to_word JSONB NOT NULL,
          link_note TEXT NOT NULL DEFAULT '',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          PRIMARY KEY (customer_id, id)
        )
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS text_connections_customer_idx
        ON text_connections (customer_id, updated_at DESC)
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
          transcript_segments JSONB NOT NULL DEFAULT '[]',
          chapters JSONB NOT NULL DEFAULT '[]',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          PRIMARY KEY (customer_id, work_id)
        )
        """)

        cur.execute("""
        ALTER TABLE podcasts
        ADD COLUMN IF NOT EXISTS transcript_segments JSONB NOT NULL DEFAULT '[]'
        """)

        cur.execute("""
        ALTER TABLE podcasts
        ADD COLUMN IF NOT EXISTS chapters JSONB NOT NULL DEFAULT '[]'
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS podcast_preferences (
          customer_id TEXT PRIMARY KEY,
          target_minutes INT NOT NULL DEFAULT 10,
          tone TEXT NOT NULL DEFAULT 'conversational',
          depth TEXT NOT NULL DEFAULT 'balanced',
          voice_a TEXT NOT NULL DEFAULT 'alloy',
          voice_b TEXT NOT NULL DEFAULT 'nova',
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS podcast_feedback (
          id SERIAL PRIMARY KEY,
          customer_id TEXT NOT NULL,
          work_id TEXT NOT NULL,
          feedback_key TEXT NOT NULL,
          note TEXT NOT NULL DEFAULT '',
          created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS podcast_feedback_customer_idx
        ON podcast_feedback (customer_id, created_at DESC)
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

        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_saved_answers (
          customer_id TEXT NOT NULL,
          id TEXT NOT NULL,
          prompt_type TEXT NOT NULL,
          work_id TEXT,
          work_title TEXT,
          section_id TEXT,
          section_label TEXT,
          token_id TEXT,
          token_text TEXT,
          associated_text TEXT NOT NULL,
          prompt_text TEXT,
          answer_text TEXT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          PRIMARY KEY (customer_id, id)
        )
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS ai_saved_answers_customer_idx
        ON ai_saved_answers (customer_id, created_at DESC)
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

    try:
        subs = stripe.Subscription.list(
            customer=customer_id,
            status="active",
            limit=1
        )
        return len(subs.data) > 0
    except Exception as e:
        print(f"Stripe check failed in has_pro for customer {customer_id}: {e}")
        return False


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


class MemorizeJudgeRequest(BaseModel):
    expected_text: str
    user_answer: str
    mode: Optional[str] = "original"


class WorkAnnotationToken(BaseModel):
    id: str
    text: str


class WorkAnnotationSection(BaseModel):
    id: Optional[str] = ""
    label: Optional[str] = ""
    text: str
    translation: Optional[str] = ""
    tokens: List[WorkAnnotationToken] = []


class AnnotateWorkRequest(BaseModel):
    work_id: Optional[str] = ""
    work_title: str
    author: Optional[str] = ""
    meta: Optional[str] = ""
    sections: List[WorkAnnotationSection] = []
    prompt_memory: Optional[str] = ""


class LinkWord(BaseModel):
    work_id: Optional[str] = ""
    work_title: Optional[str] = ""
    section_id: Optional[str] = ""
    section_label: Optional[str] = ""
    token_id: Optional[str] = ""
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


class SaveTextConnection(BaseModel):
    id: Optional[str] = ""
    from_word: LinkWord = Field(..., alias="from")
    to_word: LinkWord = Field(..., alias="to")
    link_note: Optional[str] = ""


class SaveAnnotation(BaseModel):
    work_id: str
    section_id: str
    token_id: Optional[str] = None
    content: str
    visibility: Optional[str] = "private"


class SavePhraseAnnotation(BaseModel):
    id: Optional[str] = ""
    work_id: str
    section_id: str
    token_ids: List[str] = []
    phrase_text: Optional[str] = ""
    meaning_text: str


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
    tone: Optional[str] = "conversational"
    depth: Optional[str] = "balanced"
    learner_context: Optional[List[dict]] = []


class SavePodcastPreferencesRequest(BaseModel):
    target_minutes: Optional[int] = 10
    tone: Optional[str] = "conversational"
    depth: Optional[str] = "balanced"
    voice_a: Optional[str] = "alloy"
    voice_b: Optional[str] = "nova"


class SavePodcastFeedbackRequest(BaseModel):
    work_id: str
    feedback_key: str
    note: Optional[str] = ""


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


class SaveAIAnswerRequest(BaseModel):
    id: Optional[str] = ""
    prompt_type: str
    work_id: Optional[str] = ""
    work_title: Optional[str] = ""
    section_id: Optional[str] = ""
    section_label: Optional[str] = ""
    token_id: Optional[str] = ""
    token_text: Optional[str] = ""
    associated_text: str
    prompt_text: Optional[str] = ""
    answer_text: str


class EmailCorpusRequest(BaseModel):
    email: str
    subject: Optional[str] = "Your Lexikon personal corpus"
    markdown: str


# -------------------------
# PODCAST HELPERS
# -------------------------

MAX_WORK_ANNOTATE_SECTIONS = 36
MAX_WORK_ANNOTATE_SECTION_CHARS = 24000
MIN_WORK_ANNOTATIONS = 8
MAX_WORK_ANNOTATIONS = 14
MAX_PODCAST_SOURCE_CHARS = 30000
MIN_PODCAST_MINUTES = 4
MAX_PODCAST_MINUTES = 25


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


def sample_token_ids(token_rows: List[dict], limit: int = 24) -> List[str]:
    """Pick token ids across the section (start/middle/end), not only from the beginning."""
    if not token_rows:
        return []
    ids = [str(t.get("id") or "").strip() for t in token_rows if str(t.get("id") or "").strip()]
    if len(ids) <= limit:
        return ids

    picked = []
    n = len(ids)
    for i in range(limit):
        idx = round(i * (n - 1) / max(1, (limit - 1)))
        picked.append(ids[idx])

    # Preserve order while dropping duplicates from rounding collisions.
    seen = set()
    out = []
    for token_id in picked:
        if token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
    return out


def normalize_podcast_minutes(value: Optional[int]) -> int:
    minutes = value or 10
    return max(MIN_PODCAST_MINUTES, min(MAX_PODCAST_MINUTES, int(minutes)))


def normalize_podcast_tone(value: str | None) -> str:
    tone = (value or "conversational").strip().lower()
    allowed = {"conversational", "dramatic", "scholarly"}
    return tone if tone in allowed else "conversational"


def normalize_podcast_depth(value: str | None) -> str:
    depth = (value or "balanced").strip().lower()
    allowed = {"intro", "balanced", "advanced"}
    return depth if depth in allowed else "balanced"


def build_learner_context_text(context_rows: Optional[List[dict]]) -> str:
    rows = context_rows or []
    out = []
    for row in rows[:10]:
        token = str(row.get("token") or "").strip()
        lemma = str(row.get("lemma") or "").strip()
        note = str(row.get("note") or row.get("body") or "").strip()
        if not token and not note:
            continue
        bit = f"token={token}" if token else "token=unknown"
        if lemma:
            bit += f", lemma={lemma}"
        if note:
            bit += f", note={note[:220]}"
        out.append(f"- {bit}")
    return "\n".join(out) if out else "- none provided"


def estimate_speech_seconds(text: str) -> float:
    words = max(1, len(re.findall(r"\w+", text or "")))
    # ~2.6 w/s for clear educational narration.
    return max(2.0, words / 2.6)


def build_podcast_chapters(sections: List[dict], total_seconds: float) -> List[dict]:
    cleaned = []
    for sec in sections or []:
        label = str(sec.get("label") or sec.get("id") or "").strip()
        text = str(sec.get("text") or "").strip()
        if not text:
            continue
        cleaned.append({
            "label": label or "Section",
            "weight": max(1, len(text))
        })

    if not cleaned:
        return [{"label": "Episode", "start_seconds": 0}]

    total_weight = sum(c["weight"] for c in cleaned) or 1
    t = 0.0
    chapters = []
    for row in cleaned:
        chapters.append({
            "label": row["label"],
            "start_seconds": int(round(t))
        })
        t += (row["weight"] / total_weight) * max(total_seconds, 1)
    return chapters


def count_script_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z\u0370-\u03FF\u1F00-\u1FFF0-9']+", text or ""))


def sanitize_podcast_script(script: str) -> str:
    lines = []
    last_speaker = "A"
    for raw in (script or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("HOST A:") or line.startswith("HOST B:"):
            lines.append(line)
            last_speaker = "A" if line.startswith("HOST A:") else "B"
            continue
        # Keep lines parseable for downstream speaker splitting.
        lines.append(f"HOST {last_speaker}: {line}")
    return "\n".join(lines).strip()


def generate_podcast_script(req: PodcastGenerateRequest, source_text: str) -> str:
    """Generate a two-host podcast script with a natural spoken rhythm."""
    target_minutes = normalize_podcast_minutes(req.target_minutes)
    tone = normalize_podcast_tone(req.tone)
    depth = normalize_podcast_depth(req.depth)
    min_words = int(target_minutes * 125)
    target_words = int(target_minutes * 145)
    learner_context = build_learner_context_text(req.learner_context)
    section_labels = [
        str(sec.get("label") or sec.get("id") or "").strip()
        for sec in (req.sections or [])
        if (sec.get("text") or "").strip()
    ]
    section_list = ", ".join([s for s in section_labels if s][:12]) or "none"
    prompt = f"""
Create a natural-sounding educational podcast dialogue between two hosts.
Use explicit speaker tags at the start of each line, exactly "HOST A:" or "HOST B:".
No bullet points. Keep turn lengths varied. Include occasional short interruptions and callbacks.
Never fabricate details beyond the source excerpt.

Episode settings:
- target minutes: {target_minutes}
- tone: {tone}
- depth: {depth}
- likely section labels: {section_list}

Required episode structure:
1) Hook (first ~30 seconds)
2) Historical/literary context
3) 2-3 close-reading moments with VERY short quotes and immediate paraphrase
4) Concise takeaway + "what to read next"

Style constraints:
- Sound spoken, not essay-like.
- Reference section labels naturally when relevant.
- If source is partial, explicitly say so.
- Keep quotes very short.
- Target roughly {target_words} words. Do not go below {min_words} words.

Learner context (prior struggles/notes):
{learner_context}

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
        temperature=0.65,
        max_tokens=min(4200, max(2400, target_minutes * 220)),
    )

    script = sanitize_podcast_script(r.choices[0].message.content.strip())
    words = count_script_words(script)
    if words >= min_words:
        return script

    deficit = max(180, min_words - words)
    expand_prompt = f"""
Continue this exact episode in the same voice and structure.
Write only new dialogue lines, each starting with HOST A: or HOST B:.
Do not repeat previous points. Add the missing depth and examples.
Add about {deficit} more words.

Existing script:
{script}
"""
    r2 = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classical philologist and podcast writer."},
            {"role": "user", "content": expand_prompt}
        ],
        temperature=0.55,
        max_tokens=min(2400, max(900, deficit * 2)),
    )
    continuation = sanitize_podcast_script(r2.choices[0].message.content.strip())
    joined = f"{script}\n{continuation}".strip()
    return sanitize_podcast_script(joined)


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
    if not segments:
        return []

    merged = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] = f'{merged[-1]["text"]} {seg["text"]}'.strip()
        else:
            merged.append(seg)
    return merged


def generate_podcast_audio(script: str, voice: str = "alloy", model: str = "tts-1", voices: Optional[List[str]] = None):
    """Generate TTS audio bytes and estimated transcript timings."""
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
    transcript_segments: List[Dict[str, Any]] = []
    part_idx = 0
    cursor_seconds = 0.0
    for seg in segments:
        chosen_voice = voice_a if seg["speaker"] == "A" else voice_b
        chunks = split_text_for_tts(seg["text"], max_chars=3800)
        seg_start = cursor_seconds
        seg_duration = 0.0
        for chunk in chunks:
            part_idx += 1
            safe_chunk = chunk.strip()
            if safe_chunk and safe_chunk[-1] not in ".!?":
                safe_chunk = f"{safe_chunk}."
            payload = {
                "model": model,
                "voice": chosen_voice,
                "input": safe_chunk,
                "response_format": "mp3"
            }
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if not res.ok:
                raise HTTPException(status_code=500, detail=f"TTS failed (part {part_idx}): {res.text}")
            audio_parts.append(res.content)
            chunk_seconds = estimate_speech_seconds(safe_chunk)
            seg_duration += chunk_seconds
            cursor_seconds += chunk_seconds

        transcript_segments.append({
            "speaker": seg["speaker"],
            "text": seg["text"],
            "start_seconds": round(seg_start, 2),
            "end_seconds": round(seg_start + seg_duration, 2)
        })

    return (
        b"".join(audio_parts),
        "audio/mpeg",
        ",".join([voice_a, voice_b]),
        model,
        transcript_segments
    )


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


def send_corpus_email(to_email: str, subject: str, markdown: str):
    safe_subject = html_escape(subject or "Your Lexikon personal corpus")
    safe_md = html_escape(markdown or "")
    html = f"""
        <p>Here is your exported Lexikon personal corpus.</p>
        <pre style="white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, monospace;line-height:1.45;border:1px solid #ddd;border-radius:8px;padding:12px;">{safe_md}</pre>
        <p style="color:#666;font-size:12px;">Exported from Lexikon.</p>
    """
    text = markdown or ""

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
                "subject": safe_subject,
                "html": html,
                "text": text,
            },
            timeout=8,
        )

        if response.ok:
            print(f"✅ Corpus email sent to {to_email}")
        else:
            print(f"❌ Resend API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=502, detail="Email failed")

    except Exception as e:
        print(f"❌ Failed to send corpus email: {e}")
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


@app.post("/email/corpus")
def email_corpus(payload: EmailCorpusRequest):
    email = (payload.email or "").strip().lower()
    markdown = (payload.markdown or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    if not markdown:
        raise HTTPException(status_code=400, detail="Nothing to export")

    with get_db() as cur:
        cur.execute("""
            INSERT INTO email_list (email, source, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (email) DO UPDATE
            SET source = EXCLUDED.source,
                updated_at = NOW()
        """, (email, "export:corpus"))

    send_corpus_email(email, payload.subject or "Your Lexikon personal corpus", markdown)
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
        allow_promotion_codes=True,
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

@app.get("/ai/saved-answers")
def list_saved_ai_answers(request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"answers": []}

    with get_db() as cur:
        cur.execute("""
        SELECT id, prompt_type, work_id, work_title, section_id, section_label,
               token_id, token_text, associated_text, prompt_text, answer_text, created_at
        FROM ai_saved_answers
        WHERE customer_id = %s
        ORDER BY created_at DESC
        LIMIT 500
        """, (customer_id,))
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            item = dict(r)
            if item.get("created_at"):
                item["created_at"] = item["created_at"].isoformat()
            out.append(item)
        return {"answers": out}


@app.post("/ai/saved-answers")
def save_ai_answer(req: SaveAIAnswerRequest, request: Request):
    customer_id = require_pro_user(request)
    answer_text = (req.answer_text or "").strip()
    associated_text = (req.associated_text or "").strip()
    prompt_type = (req.prompt_type or "").strip().lower()
    if not answer_text:
        raise HTTPException(status_code=400, detail="answer_text required")
    if not associated_text:
        raise HTTPException(status_code=400, detail="associated_text required")
    if not prompt_type:
        raise HTTPException(status_code=400, detail="prompt_type required")

    item_id = (req.id or "").strip() or secrets.token_urlsafe(12)
    with get_db() as cur:
        cur.execute("""
        INSERT INTO ai_saved_answers (
          customer_id, id, prompt_type, work_id, work_title, section_id, section_label,
          token_id, token_text, associated_text, prompt_text, answer_text
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id, id)
        DO UPDATE SET
          prompt_type = EXCLUDED.prompt_type,
          work_id = EXCLUDED.work_id,
          work_title = EXCLUDED.work_title,
          section_id = EXCLUDED.section_id,
          section_label = EXCLUDED.section_label,
          token_id = EXCLUDED.token_id,
          token_text = EXCLUDED.token_text,
          associated_text = EXCLUDED.associated_text,
          prompt_text = EXCLUDED.prompt_text,
          answer_text = EXCLUDED.answer_text
        """, (
            customer_id,
            item_id,
            prompt_type,
            req.work_id or None,
            req.work_title or None,
            req.section_id or None,
            req.section_label or None,
            req.token_id or None,
            req.token_text or None,
            associated_text,
            req.prompt_text or None,
            answer_text
        ))

    return {"ok": True, "id": item_id}


@app.delete("/ai/saved-answers/{answer_id}")
def delete_saved_ai_answer(answer_id: str, request: Request):
    customer_id = require_pro_user(request)
    target_id = (answer_id or "").strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="answer_id required")

    with get_db() as cur:
        cur.execute("""
        DELETE FROM ai_saved_answers
        WHERE customer_id = %s AND id = %s
        """, (customer_id, target_id))
    return {"ok": True, "id": target_id}


@app.get("/ai/credits")
def ai_credits(request: Request):
    pro = request.headers.get("X-Pro-Token")
    if has_pro(pro):
        return {"pro": True, "remaining_credits": None, "limit": None}

    remaining = get_free_ai_remaining(request)
    return {"pro": False, "remaining_credits": remaining, "limit": FREE_AI_CREDITS}


@app.post("/ai/memorize-judge")
def memorize_judge(req: MemorizeJudgeRequest, request: Request):
    try:
        expected_text = (req.expected_text or "").strip()
        user_answer = (req.user_answer or "").strip()
        mode = (req.mode or "original").strip() or "original"

        if not expected_text:
            raise HTTPException(status_code=400, detail="expected_text required")
        if not user_answer:
            raise HTTPException(status_code=400, detail="user_answer required")

        pro = request.headers.get("X-Pro-Token")
        remaining = None
        if not has_pro(pro):
            remaining = consume_free_ai_credit(request)

        prompt = f"""
You are grading a memorisation exercise with a lenient policy.
Primary goal: reward meaning-level recall, not exact wording.

Mark as CORRECT when the student's answer keeps the same core meaning, even if:
- wording is different (paraphrase/synonyms)
- clauses are reordered
- tense/case/inflection/article/pronoun choices differ
- minor details are omitted but the central proposition is intact
- punctuation/spelling is imperfect

Mark as INCORRECT only when there is a substantive meaning error, such as:
- opposite/contradictory polarity
- wrong subject/object or swapped agent/patient
- different event or fabricated key claim
- omission of a detail that changes the core meaning

If borderline or uncertain, prefer CORRECT.

Mode: {mode}
Expected:
{expected_text}

Student answer:
{user_answer}

Return strict JSON only with these keys:
- verdict: "correct" or "incorrect"
- confidence: number from 0 to 1
- similarity: number from 0 to 1 (meaning equivalence estimate)
- reason: short one-sentence explanation (max 20 words)
"""

        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict semantic grader."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=180,
        )

        raw = (r.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = {}

        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("correct", "incorrect"):
            verdict = "correct" if "correct" in raw.lower() else "incorrect"

        confidence = parsed.get("confidence", None)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        similarity = parsed.get("similarity", None)
        try:
            similarity = float(similarity)
        except Exception:
            similarity = confidence
        similarity = max(0.0, min(1.0, similarity))

        # Lenient override: if model says "incorrect" but semantic similarity is moderate,
        # treat as correct to avoid over-penalizing close answers.
        if verdict == "incorrect" and similarity >= 0.62 and confidence <= 0.9:
            verdict = "correct"

        reason = str(parsed.get("reason", "")).strip()
        if not reason:
            reason = "Meaning judged equivalent." if verdict == "correct" else "Meaning differs from the expected sentence."

        ok = verdict == "correct"
        return {
            "ok": ok,
            "verdict": verdict,
            "confidence": confidence,
            "similarity": similarity,
            "reason": reason,
            "remaining_credits": remaining,
            "limit": FREE_AI_CREDITS
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print("AI ERROR (memorize judge):", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


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


@app.post("/ai/annotate-work")
def annotate_work(req: AnnotateWorkRequest, request: Request):
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
Use this as supporting context when helpful, but prioritize the source text.
"""

        normalized_sections = []
        allowed_tokens_by_section = {}
        token_position_by_section = {}
        for sec in req.sections[:MAX_WORK_ANNOTATE_SECTIONS]:
            text = (sec.text or "").strip()
            if not text:
                continue
            section_id = (sec.id or "").strip()
            normalized_tokens = []
            for tok in sec.tokens:
                token_id = (tok.id or "").strip()
                token_text = (tok.text or "").strip()
                if not token_id or not token_text:
                    continue
                normalized_tokens.append({
                    "id": token_id,
                    "text": token_text[:64]
                })
            allowed_tokens_by_section[section_id] = {t["id"] for t in normalized_tokens}
            token_position_by_section[section_id] = {
                t["id"]: idx for idx, t in enumerate(normalized_tokens)
            }
            normalized_sections.append({
                "id": section_id,
                "label": (sec.label or sec.id or "").strip(),
                "text": text[:MAX_WORK_ANNOTATE_SECTION_CHARS],
                "translation": (sec.translation or "").strip()[:MAX_WORK_ANNOTATE_SECTION_CHARS],
                "tokens": normalized_tokens
            })

        if not normalized_sections:
            raise HTTPException(status_code=400, detail="sections required")

        section_blocks = []
        section_order = {}
        for sec in normalized_sections:
            section_order[sec["id"]] = len(section_order)
            token_lines = "\n".join([
                f'- {tok["id"]}: {tok["text"]}'
                for tok in sec["tokens"]
            ]) or "- (no tokens supplied)"
            section_blocks.append(
                f"""[{sec["label"] or sec["id"] or "section"}]
section_id: {sec["id"]}
Original:
{sec["text"]}
Translation:
{sec["translation"]}
Allowed tokens for annotations:
{token_lines}"""
            )
        sections_text = "\n\n".join(section_blocks)

        prompt = f"""
You are a classical philologist and writing coach.
Generate compact style/content annotations and map each one to a token id.

Output format:
- Return strict JSON only, no markdown, no prose outside JSON.
- JSON schema:
  {{
    "overview": ["...", "...", "..."],
    "annotations": [
      {{
        "section_id": "exact section_id from input",
        "token_id": "exact token id from that section's allowed tokens",
        "note": "short annotation for style/content reading"
      }}
    ]
  }}

Rules:
- Return {MIN_WORK_ANNOTATIONS}-{MAX_WORK_ANNOTATIONS} annotations total (if enough valid sections/tokens exist).
- Prioritize only the strongest, highest-signal observations.
- For each section, distribute annotations across beginning, middle, and end when possible; avoid clustering only in the opening lines.
- Each note under 28 words.
- Notes should mention style and/or content insight useful for reading.
- Use only provided section_id/token_id values.
- Be specific to supplied text; no invented citations.

Work title: {req.work_title}
Author: {req.author}
Meta: {req.meta}
Sections provided: {len(normalized_sections)}

{sections_text}
{memory_block}
"""

        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise, concise classical philologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        raw = (r.choices[0].message.content or "").strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None

        overview = []
        annotations = []
        def score_annotation_note(note: str) -> int:
            words = re.findall(r"[A-Za-z\u0370-\u03FF\u1F00-\u1FFF']+", note.lower())
            unique_words = {w for w in words if len(w) >= 4}
            signal_keywords = (
                "diction", "syntax", "imagery", "tone", "rhythm", "register",
                "voice", "motif", "contrast", "structure", "argument", "framing"
            )
            keyword_hits = sum(1 for k in signal_keywords if k in note.lower())
            word_count = len(note.split())
            length_score = 2 if 8 <= word_count <= 24 else 1
            return (keyword_hits * 3) + min(len(unique_words), 6) + length_score

        def trim_annotations(rows: List[dict]) -> List[dict]:
            if not rows:
                return []
            deduped = []
            seen = set()
            for item in rows:
                key = (
                    str(item.get("section_id") or "").strip(),
                    str(item.get("token_id") or "").strip(),
                    str(item.get("note") or "").strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            def rank_tuple(item: dict, idx: int):
                return (
                    score_annotation_note(str(item.get("note") or "")),
                    -section_order.get(str(item.get("section_id") or ""), 10_000),
                    -idx
                )

            if len(deduped) <= MAX_WORK_ANNOTATIONS:
                return deduped

            # Prefer coverage diversity across section thirds before pure score trimming.
            per_bucket = {}
            for idx, item in enumerate(deduped):
                sid = str(item.get("section_id") or "")
                tid = str(item.get("token_id") or "")
                positions = token_position_by_section.get(sid) or {}
                pos = positions.get(tid, 0)
                total = max(1, len(positions))
                ratio = pos / total
                bucket = 0 if ratio < 0.34 else (1 if ratio < 0.67 else 2)
                key = (sid, bucket)
                per_bucket.setdefault(key, []).append((rank_tuple(item, idx), item))

            selected = []
            picked = set()
            for key, rows_for_bucket in sorted(per_bucket.items(), key=lambda x: (section_order.get(x[0][0], 10_000), x[0][1])):
                rows_for_bucket.sort(reverse=True, key=lambda x: x[0])
                best = rows_for_bucket[0][1] if rows_for_bucket else None
                if not best:
                    continue
                uniq = (
                    str(best.get("section_id") or "").strip(),
                    str(best.get("token_id") or "").strip(),
                    str(best.get("note") or "").strip().lower(),
                )
                if uniq in picked:
                    continue
                selected.append(best)
                picked.add(uniq)
                if len(selected) >= MAX_WORK_ANNOTATIONS:
                    break

            if len(selected) < MAX_WORK_ANNOTATIONS:
                ranked_all = []
                for idx, item in enumerate(deduped):
                    uniq = (
                        str(item.get("section_id") or "").strip(),
                        str(item.get("token_id") or "").strip(),
                        str(item.get("note") or "").strip().lower(),
                    )
                    if uniq in picked:
                        continue
                    ranked_all.append((rank_tuple(item, idx), item))
                ranked_all.sort(reverse=True, key=lambda x: x[0])
                for _, item in ranked_all:
                    selected.append(item)
                    if len(selected) >= MAX_WORK_ANNOTATIONS:
                        break

            return selected[:MAX_WORK_ANNOTATIONS]

        if isinstance(parsed, dict):
            ov = parsed.get("overview")
            if isinstance(ov, list):
                overview = [str(x).strip() for x in ov if str(x).strip()][:6]

            rows = parsed.get("annotations")
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    section_id = str(item.get("section_id") or "").strip()
                    token_id = str(item.get("token_id") or "").strip()
                    note = str(item.get("note") or "").strip()
                    if not section_id or not token_id or not note:
                        continue
                    allowed = allowed_tokens_by_section.get(section_id) or set()
                    if token_id not in allowed:
                        continue
                    annotations.append({
                        "section_id": section_id,
                        "token_id": token_id,
                        "note": note[:320]
                    })

        if not annotations:
            fallback_blocks = []
            for sec in normalized_sections:
                token_ids = sample_token_ids(sec["tokens"], limit=24)
                if not token_ids:
                    continue
                fallback_blocks.append(
                    f'SECTION {sec["id"]}\nTOKENS: {", ".join(token_ids)}\nTEXT: {sec["text"][:220]}'
                )

            if fallback_blocks:
                fallback_prompt = f"""
Produce token-linked annotations as plain lines.
Format each line exactly:
section_id|token_id|note

Rules:
- Return {MIN_WORK_ANNOTATIONS}-{MAX_WORK_ANNOTATIONS} lines total (or fewer only if fewer valid sections/tokens exist).
- Focus on strongest reading cues, not exhaustive coverage.
- Spread lines across beginning/middle/end token ranges for each section when possible.
- token_id must be from the TOKENS list for that section.
- note max 24 words and about style/content reading cues.
- No extra text before or after lines.

{chr(10).join(fallback_blocks)}
"""
                try:
                    rf = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a precise classical philologist."},
                            {"role": "user", "content": fallback_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=500,
                    )
                    raw_lines = (rf.choices[0].message.content or "").strip().splitlines()
                    for line in raw_lines:
                        parts = [p.strip() for p in line.split("|", 2)]
                        if len(parts) != 3:
                            continue
                        section_id, token_id, note = parts
                        if not section_id or not token_id or not note:
                            continue
                        allowed = allowed_tokens_by_section.get(section_id) or set()
                        if token_id not in allowed:
                            continue
                        annotations.append({
                            "section_id": section_id,
                            "token_id": token_id,
                            "note": note[:320]
                        })
                except Exception as e:
                    print("AI fallback annotate parse failed:", e)

        if not annotations:
            fallback_sections = [sec for sec in normalized_sections if sec["tokens"]]
            fallback_limit = min(MAX_WORK_ANNOTATIONS, max(1, len(fallback_sections)))
            selected_sections = fallback_sections[:fallback_limit]
            for sec in selected_sections:
                section_id = sec["id"]
                first_token = next(
                    (
                        tok for tok in sec["tokens"]
                        if re.search(r"[A-Za-z\u0370-\u03FF\u1F00-\u1FFF]", tok["text"])
                    ),
                    None
                )
                if not first_token:
                    continue
                annotations.append({
                    "section_id": section_id,
                    "token_id": first_token["id"],
                    "note": "Track the section's governing claim and repeated diction; style cues reinforce the central content movement."
                })

        annotations = trim_annotations(annotations)

        if not overview:
            overview = ["AI auto-annotation generated."]

        if annotations:
            explanation = "Overview:\n" + "\n".join([f"- {line}" for line in overview])
            explanation += f"\n\nSaved annotation candidates: {len(annotations)}"
        else:
            explanation = "Overview:\n" + "\n".join([f"- {line}" for line in overview])
            explanation += "\n\nNo token-level annotations were generated."

        return {
            "explanation": explanation,
            "annotations": annotations,
            "remaining_credits": remaining,
            "limit": FREE_AI_CREDITS,
            "sections_analyzed": len(normalized_sections),
            "sections_total": len(req.sections)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("AI ERROR (annotate-work):", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# -------------------------
# PODCAST ENDPOINTS
# -------------------------

def get_podcast_preferences_row(customer_id: str) -> Dict[str, Any]:
    with get_db() as cur:
        cur.execute("""
        SELECT target_minutes, tone, depth, voice_a, voice_b, updated_at
        FROM podcast_preferences
        WHERE customer_id = %s
        """, (customer_id,))
        row = cur.fetchone()

    if not row:
        return {
            "target_minutes": 10,
            "tone": "conversational",
            "depth": "balanced",
            "voice_a": "alloy",
            "voice_b": "nova",
            "updated_at": None
        }

    return {
        "target_minutes": normalize_podcast_minutes(row.get("target_minutes")),
        "tone": normalize_podcast_tone(row.get("tone")),
        "depth": normalize_podcast_depth(row.get("depth")),
        "voice_a": row.get("voice_a") or "alloy",
        "voice_b": row.get("voice_b") or "nova",
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None
    }


@app.get("/podcasts/preferences")
def get_podcast_preferences(request: Request):
    customer_id = require_pro_user(request)
    return {"ok": True, "preferences": get_podcast_preferences_row(customer_id)}


@app.post("/podcasts/preferences")
def save_podcast_preferences(req: SavePodcastPreferencesRequest, request: Request):
    customer_id = require_pro_user(request)
    minutes = normalize_podcast_minutes(req.target_minutes)
    tone = normalize_podcast_tone(req.tone)
    depth = normalize_podcast_depth(req.depth)
    voice_a = (req.voice_a or "alloy").strip() or "alloy"
    voice_b = (req.voice_b or "nova").strip() or "nova"

    with get_db() as cur:
        cur.execute("""
        INSERT INTO podcast_preferences (customer_id, target_minutes, tone, depth, voice_a, voice_b)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id) DO UPDATE SET
          target_minutes = EXCLUDED.target_minutes,
          tone = EXCLUDED.tone,
          depth = EXCLUDED.depth,
          voice_a = EXCLUDED.voice_a,
          voice_b = EXCLUDED.voice_b,
          updated_at = NOW()
        """, (customer_id, minutes, tone, depth, voice_a, voice_b))

    return {"ok": True, "preferences": get_podcast_preferences_row(customer_id)}


@app.post("/podcasts/feedback")
def save_podcast_feedback(req: SavePodcastFeedbackRequest, request: Request):
    customer_id = require_pro_user(request)
    key = (req.feedback_key or "").strip().lower()
    allowed = {"too_basic", "too_dense", "too_long", "voice_mismatch", "great"}
    if key not in allowed:
        raise HTTPException(status_code=400, detail="Invalid feedback key")

    with get_db() as cur:
        cur.execute("""
        INSERT INTO podcast_feedback (customer_id, work_id, feedback_key, note)
        VALUES (%s, %s, %s, %s)
        """, (customer_id, req.work_id, key, (req.note or "")[:500]))

    return {"ok": True}


@app.get("/podcasts/all")
def get_all_podcasts(request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"podcasts": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT work_id, work_title, audio_mime, voice, model, chapters, created_at, updated_at
            FROM podcasts
            WHERE customer_id = %s
            ORDER BY updated_at DESC
            """, (customer_id,))
            rows = cur.fetchall() or []

        base_url = str(request.base_url).rstrip("/")
        out = []
        for row in rows:
            raw_work_id = str(row.get("work_id") or "")
            work_id_for_url = quote(raw_work_id, safe="")
            parent_work_id = raw_work_id
            section_id = ""
            is_section = False
            if "::section::" in raw_work_id:
                left, right = raw_work_id.split("::section::", 1)
                parent_work_id = left
                section_id = right
                is_section = True
            out.append({
                "work_id": raw_work_id,
                "parent_work_id": parent_work_id,
                "section_id": section_id,
                "is_section": is_section,
                "work_title": row.get("work_title") or "",
                "audio_url": f"{base_url}/podcasts/{work_id_for_url}/audio",
                "audio_mime": row.get("audio_mime") or "audio/mpeg",
                "voice": row.get("voice") or "",
                "model": row.get("model") or "",
                "chapters": row.get("chapters") or [],
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None
            })
        return {"podcasts": out}
    except Exception as e:
        print(f"Error fetching all podcasts: {e}")
        return {"podcasts": []}


@app.get("/podcasts/{work_id}")
def get_podcast_meta(work_id: str, request: Request):
    customer_id = require_pro_user(request)
    with get_db() as cur:
        cur.execute("""
        SELECT work_id, created_at, updated_at, audio_mime, transcript_segments, chapters, voice, model
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
        "transcript_segments": row.get("transcript_segments") or [],
        "chapters": row.get("chapters") or [],
        "voice": row.get("voice") or "",
        "model": row.get("model") or "",
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

    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header and range_header.startswith("bytes=") and size > 0:
        try:
            spec = range_header.split("=", 1)[1].strip()
            start_s, end_s = (spec.split("-", 1) + [""])[:2]

            if start_s == "":
                # suffix form: bytes=-N
                suffix_len = int(end_s)
                if suffix_len <= 0:
                    raise ValueError("Invalid suffix length")
                start = max(size - suffix_len, 0)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1

            if start < 0 or end < start or start >= size:
                raise ValueError("Invalid range bounds")

            end = min(end, size - 1)
            chunk = audio_data[start:end + 1]

            return Response(
                content=chunk,
                status_code=206,
                media_type=row.get("audio_mime") or "audio/mpeg",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Content-Length": str(len(chunk)),
                    "Cache-Control": "private, max-age=3600"
                }
            )
        except Exception:
            return Response(
                status_code=416,
                headers={
                    "Content-Range": f"bytes */{size}",
                    "Accept-Ranges": "bytes"
                }
            )

    return Response(
        content=audio_data,
        media_type=row.get("audio_mime") or "audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=3600",
            "Content-Length": str(size)
        }
    )


@app.delete("/podcasts/{work_id}")
def delete_podcast(work_id: str, request: Request):
    customer_id = require_pro_user(request)
    target_id = (work_id or "").strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="work_id required")

    with get_db() as cur:
        cur.execute("""
        DELETE FROM podcasts
        WHERE customer_id = %s AND work_id = %s
        """, (customer_id, target_id))
    return {"ok": True, "work_id": target_id}


@app.post("/podcasts/generate")
def generate_podcast(req: PodcastGenerateRequest, request: Request):
    try:
        print("🎙️ Podcast generate: start")
        customer_id = require_pro_user(request)
        print(f"🎙️ customer_id={customer_id} work_id={req.work_id}")

        if not req.sections:
            print("⚠️ No sections provided")
            raise HTTPException(status_code=400, detail="No sections provided")

        prefs = get_podcast_preferences_row(customer_id)
        req.target_minutes = normalize_podcast_minutes(req.target_minutes or prefs.get("target_minutes"))
        req.tone = normalize_podcast_tone(req.tone or prefs.get("tone"))
        req.depth = normalize_podcast_depth(req.depth or prefs.get("depth"))
        if not req.voices:
            req.voices = [prefs.get("voice_a") or "alloy", prefs.get("voice_b") or "nova"]

        source_text = build_podcast_source_text(req.sections)
        print(f"🎙️ source chars={len(source_text)} sections={len(req.sections)}")
        if not source_text:
            print("⚠️ No source text available")
            raise HTTPException(status_code=400, detail="No source text available")

        print("🎙️ generating script…")
        script = generate_podcast_script(req, source_text)
        print(f"🎙️ script length={len(script)}")

        print("🎙️ generating audio…")
        audio_bytes, audio_mime, voice, model, transcript_segments = generate_podcast_audio(
            script,
            voices=req.voices or ["alloy", "nova"]
        )
        total_seconds = 0.0
        if transcript_segments:
            total_seconds = max(float(transcript_segments[-1].get("end_seconds") or 0), 1.0)
        chapters = build_podcast_chapters(req.sections, total_seconds)
        print(f"🎙️ audio bytes={len(audio_bytes)} mime={audio_mime} voice={voice} model={model}")

        with get_db() as cur:
            print("🎙️ writing to DB…")
            cur.execute("""
            INSERT INTO podcasts (customer_id, work_id, work_title, script, audio, audio_mime, voice, model, transcript_segments, chapters)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (customer_id, work_id) DO UPDATE SET
              work_title = EXCLUDED.work_title,
              script = EXCLUDED.script,
              audio = EXCLUDED.audio,
              audio_mime = EXCLUDED.audio_mime,
              voice = EXCLUDED.voice,
              model = EXCLUDED.model,
              transcript_segments = EXCLUDED.transcript_segments,
              chapters = EXCLUDED.chapters,
              updated_at = NOW()
            """, (
                customer_id,
                req.work_id,
                req.title,
                script,
                psycopg2.Binary(audio_bytes),
                audio_mime,
                voice,
                model,
                Json(transcript_segments),
                Json(chapters)
            ))

        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/podcasts/{req.work_id}/audio"
        print(f"🎙️ done audio_url={audio_url}")
        return {
            "ok": True,
            "work_id": req.work_id,
            "audio_url": audio_url,
            "transcript_segments": transcript_segments,
            "chapters": chapters,
            "voice": voice,
            "model": model,
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


@app.post("/annotations/phrase")
def save_phrase_annotation(req: SavePhraseAnnotation, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    item_id = (req.id or "").strip() or secrets.token_urlsafe(12)
    meaning = (req.meaning_text or "").strip()

    with get_db() as cur:
        if not meaning:
            cur.execute("""
            DELETE FROM phrase_annotations
            WHERE customer_id = %s
              AND id = %s
            """, (customer_id, item_id))
            return {"deleted": True, "id": item_id}

        token_ids = [str(t).strip() for t in (req.token_ids or []) if str(t).strip()]
        cur.execute("""
        INSERT INTO phrase_annotations (
          customer_id, id, work_id, section_id, token_ids, phrase_text, meaning_text
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id, id)
        DO UPDATE SET
          work_id = EXCLUDED.work_id,
          section_id = EXCLUDED.section_id,
          token_ids = EXCLUDED.token_ids,
          phrase_text = EXCLUDED.phrase_text,
          meaning_text = EXCLUDED.meaning_text,
          updated_at = NOW()
        """, (
            customer_id,
            item_id,
            req.work_id,
            req.section_id,
            Json(token_ids),
            (req.phrase_text or "").strip(),
            meaning
        ))

    return {"ok": True, "id": item_id}


@app.get("/annotations/phrase")
def get_phrase_annotations(work_id: str, section_id: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"annotations": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT id, work_id, section_id, token_ids, phrase_text, meaning_text, created_at, updated_at
            FROM phrase_annotations
            WHERE customer_id = %s
              AND work_id = %s
              AND section_id = %s
            ORDER BY updated_at DESC
            """, (customer_id, work_id, section_id))
            rows = cur.fetchall() or []

        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "work_id": r["work_id"],
                "section_id": r["section_id"],
                "token_ids": list(r.get("token_ids") or []),
                "phrase_text": r.get("phrase_text") or "",
                "meaning_text": r.get("meaning_text") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
                "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None
            })
        return {"annotations": out}
    except Exception as e:
        print(f"Error fetching phrase annotations: {e}")
        return {"annotations": []}


@app.get("/annotations/phrase/all")
def get_all_phrase_annotations(request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"annotations": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT id, work_id, section_id, token_ids, phrase_text, meaning_text, created_at, updated_at
            FROM phrase_annotations
            WHERE customer_id = %s
            ORDER BY updated_at DESC
            """, (customer_id,))
            rows = cur.fetchall() or []

        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "work_id": r["work_id"],
                "section_id": r["section_id"],
                "token_ids": list(r.get("token_ids") or []),
                "phrase_text": r.get("phrase_text") or "",
                "meaning_text": r.get("meaning_text") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
                "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None
            })
        return {"annotations": out}
    except Exception as e:
        print(f"Error fetching all phrase annotations: {e}")
        return {"annotations": []}


@app.post("/connections")
def save_text_connection(req: SaveTextConnection, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    item_id = (req.id or "").strip() or secrets.token_urlsafe(12)
    note = (req.link_note or "").strip()
    from_payload = req.from_word.model_dump(by_alias=True)
    to_payload = req.to_word.model_dump(by_alias=True)

    with get_db() as cur:
        cur.execute("""
        INSERT INTO text_connections (customer_id, id, from_word, to_word, link_note)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (customer_id, id)
        DO UPDATE SET
          from_word = EXCLUDED.from_word,
          to_word = EXCLUDED.to_word,
          link_note = EXCLUDED.link_note,
          updated_at = NOW()
        """, (
            customer_id,
            item_id,
            Json(from_payload),
            Json(to_payload),
            note
        ))
    return {"ok": True, "id": item_id}


@app.delete("/connections/{connection_id}")
def delete_text_connection(connection_id: str, request: Request):
    customer_id = require_pro_user(request)
    target_id = (connection_id or "").strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="connection_id required")

    with get_db() as cur:
        cur.execute("""
        DELETE FROM text_connections
        WHERE customer_id = %s AND id = %s
        """, (customer_id, target_id))
    return {"ok": True, "id": target_id}


@app.get("/connections/for-section")
def get_connections_for_section(work_id: str, section_id: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"connections": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT id, from_word, to_word, link_note, created_at, updated_at
            FROM text_connections
            WHERE customer_id = %s
            ORDER BY updated_at DESC
            """, (customer_id,))
            rows = cur.fetchall() or []

        out = []
        for r in rows:
            f = r.get("from_word") or {}
            t = r.get("to_word") or {}
            if not (
                (str(f.get("work_id") or "") == str(work_id) and str(f.get("section_id") or "") == str(section_id))
                or (str(t.get("work_id") or "") == str(work_id) and str(t.get("section_id") or "") == str(section_id))
            ):
                continue
            out.append({
                "id": r.get("id"),
                "from": f,
                "to": t,
                "link_note": r.get("link_note") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
                "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None
            })
        return {"connections": out}
    except Exception as e:
        print(f"Error fetching connections for section: {e}")
        return {"connections": []}


@app.get("/connections/all")
def get_all_connections(request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        return {"connections": []}

    try:
        with get_db() as cur:
            cur.execute("""
            SELECT id, from_word, to_word, link_note, created_at, updated_at
            FROM text_connections
            WHERE customer_id = %s
            ORDER BY updated_at DESC
            """, (customer_id,))
            rows = cur.fetchall() or []

        out = []
        for r in rows:
            out.append({
                "id": r.get("id"),
                "from": r.get("from_word") or {},
                "to": r.get("to_word") or {},
                "link_note": r.get("link_note") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
                "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None
            })
        return {"connections": out}
    except Exception as e:
        print(f"Error fetching all connections: {e}")
        return {"connections": []}


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

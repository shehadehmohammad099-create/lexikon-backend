import os
import secrets
import stripe
import openai
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import Body
import requests
from contextlib import contextmanager


DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")


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
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
        )
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


class ExplainPassage(BaseModel):
    work: str
    section: str
    speaker: Optional[str] = ""
    greek: str
    translation: Optional[str] = ""


class SaveAnnotation(BaseModel):
    work_id: str
    section_id: str
    token_id: Optional[str] = None
    content: str
    visibility: Optional[str] = "private"


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
                "from": "Lexikon <onboarding@resend.dev>",
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
        success_url=f"{origin}/static/app.html?session_id={{CHECKOUT_SESSION_ID}}",
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

        restore_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)

        with get_db() as cur:
            cur.execute("""
                INSERT INTO restore_tokens (token, customer_id, expires_at)
                VALUES (%s, %s, %s)
            """, (restore_token, customer_id, expires_at))

        restore_url = f"{FRONTEND_URL}/static/app.html?restore_token={restore_token}"
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

        origin = request.headers.get("origin") or FRONTEND_URL
        restore_url = f"{origin}/static/app.html?restore_token={restore_token}"

        print("RESTORE LINK:", restore_url)
        print("CHECKOUT EMAIL:", email)

        if email:
            print(f"📧 Attempting to send restore email to {email}...")
            send_restore_email(email, restore_url)
        else:
            print("⚠️ No email found, skipping restore email")

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

    restore_url = f"{FRONTEND_URL}/static/app.html?restore_token={restore_token}"

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
        return_url=f"{origin}/static/app.html"
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

@app.post("/ai/explain-word")
def explain_word(req: ExplainWord, request: Request):
    try:
        pro = request.headers.get("X-Pro-Token")
        if not has_pro(pro):
            raise HTTPException(status_code=402, detail="Pro required")

        prompt = f"""
Explain this word philologically.
It is either in Latin or Greek.
Word: {req.token}
Lemma: {req.lemma}
POS: {req.pos}
Morphology: {req.morph}
Context: {req.sentence}
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

        return {"explanation": r.choices[0].message.content.strip()}

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
        if not has_pro(pro):
            raise HTTPException(status_code=402, detail="Pro required")

        prompt = f"""
Explain this passage philologically.
Do NOT summarize or paraphrase.
Focus on syntax, structure, and argumentative flow.
Explain why the Greek may be difficult to read.

Work: {req.work}
Section: {req.section}
Speaker: {req.speaker}

Greek text:
{req.greek}

Translation (for reference only):
{req.translation}
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

        return {"explanation": r.choices[0].message.content.strip()}

    except HTTPException as e:
        raise e
    except Exception as e:
        print("AI ERROR (passage):", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


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

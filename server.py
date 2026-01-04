import os
import secrets
import stripe
import openai
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


DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cur = conn.cursor(cursor_factory=RealDictCursor)

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



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://the-lexicon-project.netlify.app", "*"],            # for now
    allow_credentials=False,        # IMPORTANT
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ENV
# -------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
STRIPE_SECRET_KEY = os.environ["STRIPE_SECRET_KEY"]
STRIPE_PRICE_ID = os.environ["STRIPE_PRICE_ID"]
FRONTEND_URL = os.environ["FRONTEND_URL"]  # e.g. https://lexikon.netlify.app/app.html

openai.api_key = OPENAI_API_KEY
stripe.api_key = STRIPE_SECRET_KEY



# -------------------------
# TEMP PRO TOKENS (memory)
# -------------------------
PRO_TOKENS = set()

def mint_pro_token():
    t = secrets.token_urlsafe(32)
    PRO_TOKENS.add(t)
    return t

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

# -------------------------
# STRIPE
# -------------------------
    

@app.get("/create-checkout-session")
def create_checkout_session(request: Request):
    origin = request.headers.get("origin")

    if not origin:
        origin = "https://the-lexicon-project.netlify.app"

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        billing_address_collection="required",
        success_url=f"{origin}/frontend/static/app.html?session_id={{CHECKOUT_SESSION_ID}}",
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

        # mint restore token
        restore_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)

        cur.execute("""
            INSERT INTO restore_tokens (token, customer_id, expires_at)
            VALUES (%s, %s, %s)
        """, (restore_token, customer_id, expires_at))

        restore_url = f"{FRONTEND_URL}?restore_token={restore_token}"

        send_restore_email(email, restore_url)

    return {"ok": True}

def send_restore_email(to_email: str, restore_url: str):
    requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "from": "Lexikon <restore@lexikon.app>",
            "to": to_email,
            "subject": "Restore your Lexikon subscription",
            "html": f"""
                <p>You can restore your Lexikon Pro access by clicking the link below:</p>

                <p>
                <a href="{restore_url}">
                    Restore my subscription
                </a>
                </p>

                <p>This link can be used once and will expire.</p>
            """
        },
        timeout=5,
    )


@app.get("/checkout-success")
def checkout_success(session_id: str, request: Request):
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

    # Issue Pro token
    pro_token = secrets.token_urlsafe(32)
    cur.execute(
        "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
        (pro_token, customer_id)
    )

    # Restore token
    restore_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)

    cur.execute("""
        INSERT INTO restore_tokens (token, customer_id, expires_at)
        VALUES (%s, %s, %s)
    """, (restore_token, customer_id, expires_at))

    # 🔑 THIS IS THE IMPORTANT PART
    origin = request.headers.get("origin") or FRONTEND_URL

    restore_url = f"{origin}/frontend/static/app.html?restore_token={restore_token}"

    print("RESTORE LINK:", restore_url)

    return {"pro_token": pro_token}

    print("CHECKOUT EMAIL:", email)

    restore_url = f"{origin}/frontend/static/app.html?restore_token={restore_token}"

    if email:
        send_restore_email(email, restore_url)



    session = stripe.checkout.Session.retrieve(
        session_id,
        expand=["customer"]
    )

    customer_id = session.customer.id
    email = session.customer.email  # THIS is reliable here

    if not customer_id:
        raise HTTPException(status_code=400, detail="Missing customer")

    # Issue Pro token (existing behavior)
    pro_token = secrets.token_urlsafe(32)
    cur.execute(
        "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
        (pro_token, customer_id)
    )

    # 🔐 CREATE RESTORE TOKEN HERE (key change)
    restore_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)

    cur.execute("""
        INSERT INTO restore_tokens (token, customer_id, expires_at)
        VALUES (%s, %s, %s)
    """, (restore_token, customer_id, expires_at))

    restore_url = f"{FRONTEND_URL}?restore_token={restore_token}"

    print("RESTORE LINK (SAVE THIS):", restore_url)

    return {"pro_token": pro_token}
    try:
        session = stripe.checkout.Session.retrieve(session_id)

        customer_id = session.customer
        if not customer_id:
            raise HTTPException(status_code=400, detail="Missing customer")

        token = secrets.token_urlsafe(32)

        cur.execute(
            "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
            (token, customer_id)
        )

        return {"pro_token": token}

    except HTTPException as e:
        raise e

    except Exception as e:
        print("CHECKOUT ERROR:", e)
        return {"error": str(e)}



def customer_from_token(pro_token: str | None):
    if not pro_token:
        return None

    cur.execute(
        "SELECT customer_id FROM pro_tokens WHERE token = %s",
        (pro_token,)
    )

    row = cur.fetchone()
    return row["customer_id"] if row else None






# -------------------------
# AI (PAID)
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
        # preserve correct status codes like 402
        raise e

    except Exception as e:
        # THIS is what was causing the “CORS” error
        print("AI ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


class ExplainPassage(BaseModel):
    work: str
    section: str
    speaker: Optional[str] = ""
    greek: str
    translation: Optional[str] = ""

# ---- endpoint ----

@app.post("/ai/explain-passage")
def explain_passage(req: ExplainPassage, request: Request):
    try:
        # Pro check (same as explain-word)
        pro = request.headers.get("X-Pro-Token")
        if not has_pro(pro):
            raise HTTPException(status_code=402, detail="Pro required")

        # Prompt (commentary, not summary)
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

        # Same OpenAI call style as explain-word
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
            "explanation": r.choices[0].message.content.strip()
        }

    except HTTPException as e:
        # Preserve correct status codes (e.g. 402)
        raise e

    except Exception as e:
        # Prevent masked CORS errors
        print("AI ERROR (passage):", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# @app.get("/create-portal-session")
# def create_portal_session(request: Request):
#     origin = request.headers.get("origin") or "https://the-lexicon-project.netlify.app"

#     portal = stripe.billing_portal.Session.create(
#         return_url=f"{origin}/static/app.html?restore=1"
#     )

#     return {"url": portal.url}


from datetime import datetime, timedelta
# @app.post("/billing/request-restore")
# async def request_restore(request: Request):
    # print("HIT")
    # payload = await request.json()
    # email = payload.get("email")

    # if not email:
    #     # Always succeed to avoid email enumeration
    #     return {"ok": True}

    # customers = stripe.Customer.search(
    #     query=f"email:'{email}'",
    #     limit=1
    # ).data

    # if not customers:
    #     return {"ok": True}

    # customer_id = customers[0].id
    # print("CUSTOMERS FOUND:", len(customers))
    # restore_token = secrets.token_urlsafe(32)
    # expires_at = datetime.utcnow() + timedelta(minutes=15)

    # cur.execute("""
    #     INSERT INTO restore_tokens (token, customer_id, expires_at)
    #     VALUES (%s, %s, %s)
    # """, (restore_token, customer_id, expires_at))

    # restore_url = f"{FRONTEND_URL}?restore_token={restore_token}"

    # # DEV MODE: log instead of email
    # print("RESTORE LINK:", restore_url)

    # return {"ok": True}

    # email = payload.get("email")
    # if not email:
    #     raise HTTPException(status_code=400, detail="Email required")

    # # 🔹 IMPORTANT: use Stripe SEARCH (best we can do here)
    # customers = stripe.Customer.search(
    #     query=f"email:'{email}'",
    #     limit=1
    # ).data

    # if not customers:
    #     # Do NOT leak info
    #     return {"ok": True}

    # customer_id = customers[0].id

    # restore_token = secrets.token_urlsafe(32)
    # expires = datetime.utcnow() + timedelta(minutes=15)

    # cur.execute("""
    #     INSERT INTO restore_tokens (token, customer_id, expires_at)
    #     VALUES (%s, %s, %s)
    # """, (restore_token, customer_id, expires))

    # restore_url = f"{FRONTEND_URL}?restore_token={restore_token}"

    # # 🔹 SEND EMAIL HERE
    # # send_email(
    # #   to=email,
    # #   subject="Restore your Lexikon subscription",
    # #   body=f"Click to restore: {restore_url}"
    # # )

    # print("RESTORE LINK:", restore_url)  # dev-only

    # return {"ok": True}

@app.post("/billing/restore-from-link")
async def restore_from_link(request: Request):
    payload = await request.json()
    token = payload.get("restore_token")

    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

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

    # One-time use
    cur.execute("DELETE FROM restore_tokens WHERE token = %s", (token,))

    return {"pro_token": pro_token}

    token = payload.get("restore_token")
    if not token:
        raise HTTPException(status_code=400)

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

    # Ensure subscription still active
    subs = stripe.Subscription.list(
        customer=customer_id,
        status="active",
        limit=1
    ).data

    if not subs:
        raise HTTPException(status_code=402)

    pro_token = mint_pro_token(customer_id)

    # One-time use
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



    email = payload.get("email")

    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    # 1. Find Stripe customer by email
    customers = stripe.Customer.search(
        query=f"email:'{email}'",
        limit=1
    ).data

    if not customers:
        raise HTTPException(status_code=404, detail="No customer found")

    customer = customers[0]

    # 2. Check active subscription
    subs = stripe.Subscription.list(
        customer=customer.id,
        status="active",
        limit=1
    ).data

    if not subs:
        raise HTTPException(status_code=402, detail="No active subscription")

    # 3. Mint a fresh Pro token
    pro_token = mint_pro_token(customer.id)

    return {
        "pro_token": pro_token
    }




@app.get("/billing/restore-token")
def billing_restore_token(session_id: str):
    session = stripe.checkout.Session.retrieve(session_id)

    customer_id = session.customer
    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer on session")

    # Check active subscription
    subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
    if not subs.data:
        raise HTTPException(status_code=402, detail="No active subscription")

    token = secrets.token_urlsafe(32)

    cur.execute(
        "INSERT INTO pro_tokens (token, customer_id) VALUES (%s, %s)",
        (token, customer_id)
    )

    return {"pro_token": token}



class SaveAnnotation(BaseModel):
    work_id: str
    section_id: str
    token_id: Optional[str] = None  # word-level; can be null for passage-level later
    content: str
    visibility: Optional[str] = "private"

@app.post("/annotations")
def save_annotation(req: SaveAnnotation, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Upsert by unique index
    cur.execute("""
    INSERT INTO annotations (customer_id, work_id, section_id, token_id, content, visibility)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (customer_id, work_id, section_id, token_id)
    DO UPDATE SET content = EXCLUDED.content, visibility = EXCLUDED.visibility, updated_at = NOW()
    """, (customer_id, req.work_id, req.section_id, req.token_id, req.content, req.visibility))

    return {"ok": True}


@app.post("/annotations")
def save_annotation(req: SaveAnnotation, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401)

    # DELETE if empty
    if not req.content.strip():
        cur.execute("""
        DELETE FROM annotations
        WHERE customer_id = %s
          AND work_id = %s
          AND section_id = %s
          AND token_id IS NOT DISTINCT FROM %s
        """, (
            customer_id,
            req.work_id,
            req.section_id,
            req.token_id
        ))
        return {"deleted": True}

    # Otherwise UPSERT
    cur.execute("""
    INSERT INTO annotations (customer_id, work_id, section_id, token_id, content)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (customer_id, work_id, section_id, token_id)
    DO UPDATE SET content = EXCLUDED.content, updated_at = NOW()
    """, (
        customer_id,
        req.work_id,
        req.section_id,
        req.token_id,
        req.content
    ))

    return {"ok": True}

@app.get("/annotations")
def get_annotations(work_id: str, section_id: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)

    if not customer_id:
        raise HTTPException(status_code=401)

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
        out[r["token_id"]] = r["content"]

    return {"annotations": out}



@app.get("/annotations/public")
def get_public_annotations(work_id: str, section_id: str):
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
        out[r["token_id"]] = r["content"]

    return {"annotations": out}


@app.get("/annotations/search")
def search_annotations(q: str, request: Request):
    pro = request.headers.get("X-Pro-Token")
    customer_id = customer_from_token(pro)
    if not customer_id:
        raise HTTPException(status_code=401)

    cur.execute("""
    SELECT work_id, section_id, token_id, content
    FROM annotations
    WHERE customer_id = %s
      AND content ILIKE %s
    ORDER BY updated_at DESC
    LIMIT 50
    """, (customer_id, f"%{q}%"))

    rows = cur.fetchall()
    return {"results": rows}


@app.get("/restore-pro")
def restore_pro(session_id: str):
    session = stripe.checkout.Session.retrieve(session_id)
    customer_id = session.customer

    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer")

    # Check active subscription
    subs = stripe.Subscription.list(
        customer=customer_id,
        status="active",
        limit=1
    )

    if not subs.data:
        raise HTTPException(status_code=402, detail="No active subscription")

    # Issue NEW token
    token = secrets.token_urlsafe(32)

    cur.execute("""
    INSERT INTO pro_tokens (token, customer_id)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING
    """, (token, customer_id))

    return {"pro_token": token}

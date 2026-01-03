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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # for now
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

def has_pro(t):
    return t in PRO_TOKENS

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
    
FRONTEND_URL = "https://the-lexicon-project.netlify.app"
# @app.get("/create-checkout-session")
# def create_checkout_session():
#     session = stripe.checkout.Session.create(
#         mode="subscription",
#         line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
#         success_url=f"{FRONTEND_URL}/static/confirmation.html?session_id={{CHECKOUT_SESSION_ID}}",
#         cancel_url=FRONTEND_URL,
#     )
#     return {"url": session.url}

@app.get("/create-checkout-session")
def create_checkout_session(request: Request):
    origin = request.headers.get("origin")

    # Fallback (prod)
    if not origin:
        origin = "https://the-lexicon-project.netlify.app"

    # ALWAYS go to index.html
    success_url = f"{origin}/frontend/index.html?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{origin}/index.html"

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
    )

    return {"url": session.url}



@app.get("/checkout-success")
def checkout_success(session_id: str):
    session = stripe.checkout.Session.retrieve(session_id)
    if session.status != "complete":
        raise HTTPException(400, "Payment not completed")

    return {"pro_token": mint_pro_token()}

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

# --- endpoint ---

@app.post("/ai/explain-passage")
def explain_passage(req: ExplainPassage, request: Request):
    try:
        # 1. Pro check (identical to explain-word)
        pro = request.headers.get("X-Pro-Token")
        if not has_pro(pro):
            raise HTTPException(status_code=402, detail="Pro required")

        # 2. Prompt (philology-first, NOT summary)
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

        # 3. Same OpenAI call style
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
        # preserve correct status codes like 402
        raise e

    except Exception as e:
        # same defensive error handling as explain-word
        print("AI ERROR (passage):", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

import os
import re
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# ================================
# Config (env vars)
# ================================
GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", "glossary_rollomatic.csv")

SUPERTEXT_API_KEY = os.getenv("SUPERTEXT_API_KEY", "")
SUPERTEXT_ENDPOINT = os.getenv("SUPERTEXT_ENDPOINT", "https://api.supertext.com/v1/translate/ai/text")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT", "")  # e.g. https://xxxx.services.ai.azure.com/openai/v1/
MISTRAL_DEPLOYMENT = os.getenv("MISTRAL_DEPLOYMENT", "")  # e.g. mistral-medium-2505

# CORS (comma-separated origins). Example: "https://<username>.github.io"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

LANG_FULL = {"de": "German", "en": "English", "fr": "French", "it": "Italian"}

SUPERTEXT_SOURCE_MAP = {"de": "de", "en": "en", "fr": "fr", "it": "it"}
SUPERTEXT_TARGET_MAP = {"de": "de-DE", "en": "en-US", "fr": "fr-FR", "it": "it-IT"}


def get_lang_full(code: str) -> str:
    return LANG_FULL.get(code, code)


def simple_clean(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return "" if text.lower() == "nan" else text


# ================================
# Load glossary once at startup
# ================================
langs = ["fr", "en", "de", "it"]
glossary_terms: Dict[str, Dict[str, Dict[str, str]]] = {}


def load_glossary(path: str) -> None:
    global glossary_terms
    df = pd.read_csv(path)
    glossary_terms = {}
    for src_lang in langs:
        glossary_terms[src_lang] = {}
        for _, row in df.iterrows():
            src_term = simple_clean(row.get(src_lang, "")).lower()
            if src_term:
                glossary_terms[src_lang][src_term] = {
                    lg: simple_clean(row.get(lg, ""))
                    for lg in langs
                    if pd.notna(row.get(lg, None)) and simple_clean(row.get(lg, "")) != ""
                }


def extract_terms(src_lang: str, tgt_lang: str, src_text: str) -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    text = simple_clean(src_text).lower()

    if src_lang not in glossary_terms:
        return found

    for term_src, translations in glossary_terms[src_lang].items():
        pattern = r"\b" + re.escape(term_src) + r"\b"
        if re.search(pattern, text):
            if tgt_lang in translations:
                found.append({"src_term": term_src, "tgt_term": translations[tgt_lang]})
    return found


# ================================
# SuperText + Mistral
# ================================
def supertext_translate(src_lang: str, tgt_lang: str, text: str, timeout: int = 30) -> str:
    if not SUPERTEXT_API_KEY:
        raise RuntimeError("Missing SUPERTEXT_API_KEY")

    source_clean = simple_clean(text)

    payload = {
        "text": [source_clean],
        "source_lang": SUPERTEXT_SOURCE_MAP[src_lang],
        "target_lang": SUPERTEXT_TARGET_MAP[tgt_lang],
    }

    headers = {
        "Authorization": f"Supertext-Auth-Key {SUPERTEXT_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(SUPERTEXT_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"SuperText error {r.status_code}: {r.text}")

    return r.json()["translated_text"][0].strip()


def build_mistral_prompt(
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    draft: str,
    terms: List[Dict[str, str]],
) -> str:
    src_lang_full = get_lang_full(src_lang)
    tgt_lang_full = get_lang_full(tgt_lang)
    source_cleaned = simple_clean(src_text)

    if terms:
        terms_block = "\n".join(f"- {t['src_term']} → {t['tgt_term']}" for t in terms)
    else:
        terms_block = "No glossary terms for this sentence."

    prompt = f"""
You are a senior-level technical industry translator with expertise in:
- CNC grinding machines & spindle behaviour
- Carbide tool geometry & multi-axis kinematics
- Robot loaders, tool holders, clamping & collet systems
- Runout, offsets, dressing cycles & coolant management
- Drill/Endmill manufacturing, resharpening, coating
- High-precision HMI configuration for machining workflows

Your goal is to produce the best possible final translation, using:
- the original source sentence
- the preliminary draft translation
- glossary terms when they improve accuracy or terminology

Key rules:
- Glossary terms are suggestions, not mandatory substitutions.
- You may use them, adapt them, or ignore them if not appropriate in context.
- Preserve numbers, units, and symbols.
- If the draft is already optimal, return it unchanged.

Task:
Translate from {src_lang_full} to {tgt_lang_full} by improving the draft when needed.

Source text:
{source_cleaned}

Initial draft translation:
{draft}

Glossary suggestions:
{terms_block}

Output (Provide only the improved translation, with no explanation):
""".strip()

    return prompt

def refine_with_mistral(
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    draft: str,
    terms: List[Dict[str, str]],
) -> str:
    # If Mistral not configured, fallback to draft
    if not (MISTRAL_API_KEY and MISTRAL_ENDPOINT and MISTRAL_DEPLOYMENT):
        return draft

    prompt = build_mistral_prompt(src_lang, tgt_lang, src_text, draft, terms)

    url = MISTRAL_ENDPOINT.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    }

    payload = {
        "model": MISTRAL_DEPLOYMENT,
        "messages": [
            {"role": "system", "content": "You are a translation refinement engine."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 220,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            # fallback to draft on error
            return draft

        data = r.json()
        out = data["choices"][0]["message"]["content"].strip()
        return out if out else draft

    except Exception:
        return draft
        

# ================================
# FastAPI app
# ================================
app = FastAPI(title="Rollomatic Translator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else [o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    src_lang: str = Field(..., pattern="^(fr|en|de|it)$")
    tgt_lang: str = Field(..., pattern="^(fr|en|de|it)$")
    text: str = Field(..., min_length=1, max_length=10000)
    debug: bool = False


class TranslateResponse(BaseModel):
    translation: str
    draft: Optional[str] = None
    terms: Optional[List[Dict[str, str]]] = None


@app.on_event("startup")
def _startup() -> None:
    if not os.path.exists(GLOSSARY_PATH):
        raise RuntimeError(f"Glossary file not found: {GLOSSARY_PATH}")
    load_glossary(GLOSSARY_PATH)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    if req.src_lang == req.tgt_lang:
        return TranslateResponse(translation=req.text)

    try:
        terms = extract_terms(req.src_lang, req.tgt_lang, req.text)
        draft = supertext_translate(req.src_lang, req.tgt_lang, req.text)
        final = refine_with_mistral(req.src_lang, req.tgt_lang, req.text, draft, terms)

        if req.debug:
            return TranslateResponse(translation=final, draft=draft, terms=terms)
        return TranslateResponse(translation=final)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

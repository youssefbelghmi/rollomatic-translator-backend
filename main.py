import os
import re
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration (environment variables)

# Path to the bilingual/multilingual glossary used to enforce consistent terminology.
GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", "glossary_rollomatic.csv")

# SuperText machine translation service configuration.
SUPERTEXT_API_KEY = os.getenv("SUPERTEXT_API_KEY", "")
SUPERTEXT_ENDPOINT = os.getenv("SUPERTEXT_ENDPOINT", "https://api.supertext.com/v1/translate/ai/text")

# Optional Mistral refinement (post-editing) configuration.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT", "")
MISTRAL_DEPLOYMENT = os.getenv("MISTRAL_DEPLOYMENT", "")

# Simple API key check for protecting the /translate endpoint.
ACCESS_KEY = os.getenv("ACCESS_KEY", "")

# CORS (comma-separated origins).
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Human-readable language names (used in the LLM prompt).
LANG_FULL = {"de": "German", "en": "English", "fr": "French", "it": "Italian"}

# SuperText expects specific source/target identifiers.
SUPERTEXT_SOURCE_MAP = {"de": "de", "en": "en", "fr": "fr", "it": "it"}
SUPERTEXT_TARGET_MAP = {"de": "de-DE", "en": "en-US", "fr": "fr-FR", "it": "it-IT"}


def get_lang_full(code: str) -> str:
    """
    Convert a language code (e.g., 'fr') into its full English name (e.g., 'French').

    Args:
        code: ISO-like language code.

    Returns:
        Full language name if known, otherwise returns the original code.
    """
    return LANG_FULL.get(code, code)


def simple_clean(text: Any) -> str:
    """
    Normalize input to a safe, trimmed string for downstream processing.

    - Converts None to an empty string.
    - Strips surrounding whitespace.
    - Converts the literal string "nan" (case-insensitive) to empty, since it commonly
      appears when reading CSVs via pandas.

    Args:
        text: Any incoming value (possibly None, float('nan'), etc.).

    Returns:
        A cleaned string (possibly empty).
    """
    if text is None:
        return ""
    text = str(text).strip()
    return "" if text.lower() == "nan" else text


# Load glossary once at startup

# Supported languages in the glossary CSV.
langs = ["fr", "en", "de", "it"]

# IMPORTANT CHANGE:
# We now store multiple translations per (src_lang, src_term, tgt_lang).
# Structure:
# glossary_terms[src_lang][src_term_lower][tgt_lang] = [list of distinct translations]
glossary_terms: Dict[str, Dict[str, Dict[str, List[str]]]] = {}


def load_glossary(path: str) -> None:
    """
    Load the glossary CSV file into an in-memory dictionary for fast lookup.

    The CSV is expected to contain one column per language in `langs`.
    Each row represents a term and its translations across the supported languages.

    IMPORTANT:
    - If the same source term appears multiple times with different translations,
      we keep ALL distinct translations per target language.

    Args:
        path: Path to the glossary CSV file.

    Raises:
        Any exception raised by pandas while reading the CSV (caller handles).
    """
    global glossary_terms
    df = pd.read_csv(path)

    # Ensure columns exist (best effort, no hard fail; missing cols simply ignored)
    glossary_terms = {src_lang: {} for src_lang in langs}

    for _, row in df.iterrows():
        # We treat each column as potential "source language"
        for src_lang in langs:
            src_term_raw = simple_clean(row.get(src_lang, ""))
            src_term = src_term_raw.lower()

            if not src_term:
                continue

            if src_term not in glossary_terms[src_lang]:
                glossary_terms[src_lang][src_term] = {}

            # For each possible target language, accumulate distinct translations
            for tgt_lang in langs:
                if tgt_lang == src_lang:
                    continue

                tgt_val = simple_clean(row.get(tgt_lang, ""))
                if not tgt_val:
                    continue

                bucket = glossary_terms[src_lang][src_term].setdefault(tgt_lang, [])
                # Keep distinct values only (case-sensitive here after clean; you can lower() if you want)
                if tgt_val not in bucket:
                    bucket.append(tgt_val)


def extract_terms(src_lang: str, tgt_lang: str, src_text: str) -> List[Dict[str, Any]]:
    """
    Extract glossary term matches from a source text and return ALL target equivalents.

    Matching is done using word boundaries to reduce false positives (e.g., avoid matching
    'tool' inside 'tooling').

    IMPORTANT:
    - If a matched glossary term has multiple translations for the same target language,
      we return them all as "tgt_terms".

    Args:
        src_lang: Source language code.
        tgt_lang: Target language code.
        src_text: Raw source text to scan.

    Returns:
        A list of dictionaries like:
        [{"src_term": "...", "tgt_terms": ["...", "...", ...]}, ...]
        Returns an empty list if no terms are found or glossary is missing for src_lang.
    """
    found: List[Dict[str, Any]] = []
    text = simple_clean(src_text).lower()

    if src_lang not in glossary_terms:
        return found

    for term_src, translations_by_lang in glossary_terms[src_lang].items():
        pattern = r"\b" + re.escape(term_src) + r"\b"
        if re.search(pattern, text):
            tgt_terms = translations_by_lang.get(tgt_lang, [])
            if tgt_terms:
                found.append({"src_term": term_src, "tgt_terms": tgt_terms})
    return found


# SuperText + Mistral

def supertext_translate(src_lang: str, tgt_lang: str, text: str, timeout: int = 60) -> str:
    """
    Translate text using the SuperText API.

    Args:
        src_lang: Source language code (must exist in SUPERTEXT_SOURCE_MAP).
        tgt_lang: Target language code (must exist in SUPERTEXT_TARGET_MAP).
        text: Source text to translate.
        timeout: HTTP timeout in seconds.

    Returns:
        The translated string (first element returned by the API).

    Raises:
        RuntimeError: If the API key is missing or if SuperText returns a non-200 status.
        KeyError: If src_lang/tgt_lang is not present in the mapping dictionaries.
        requests.RequestException: If the request fails at the HTTP level.
    """
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
    terms: List[Dict[str, Any]],
) -> str:
    """
    Build a detailed post-editing prompt to refine a draft translation.

    The prompt provides:
    - domain context (technical industry / CNC grinding)
    - the original source text
    - the initial machine translation draft
    - glossary term suggestions (optional)

    IMPORTANT:
    - If a glossary term has multiple possible translations in tgt_lang,
      ALL of them are shown to Mistral so it can choose the best one in context.

    Args:
        src_lang: Source language code.
        tgt_lang: Target language code.
        src_text: Original text to translate.
        draft: Draft translation (typically from SuperText).
        terms: Glossary term hits extracted from src_text.

    Returns:
        A single prompt string to be sent to a chat-completion endpoint.
    """
    src_lang_full = get_lang_full(src_lang)
    tgt_lang_full = get_lang_full(tgt_lang)
    source_cleaned = simple_clean(src_text)

    if terms:
        # Show all candidates per term, e.g.:
        # - spindle → broche | fuseau
        terms_lines: List[str] = []
        for t in terms:
            src_term = t.get("src_term", "")
            tgt_terms = t.get("tgt_terms", []) or []
            # Make it readable: join with " | "
            tgt_block = " | ".join(str(x) for x in tgt_terms)
            terms_lines.append(f"- {src_term} → {tgt_block}")
        terms_block = "\n".join(terms_lines)
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
- Some glossary entries may include MULTIPLE valid target translations. Choose the best one in context.
- You may:
    - use one of the glossary options exactly as given,  
    - adapt/rephrase it for contextual correctness,  
    - ignore glossary suggestions entirely if they do not fit meaning.
- The final choice must always maximize clarity, accuracy & terminology relevance.
- Preserve numbers, units, and symbols.
- If the first draft translation is already optimal, return unchanged.

Task:
Translate from {src_lang_full} to {tgt_lang_full} by improving the draft when needed.

Source text:
{source_cleaned}

Initial draft translation:
{draft}

Glossary suggestions (may include multiple options per term):
{terms_block}

Output (Provide only the improved translation, with no explanation):
""".strip()

    return prompt


def refine_with_mistral(
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    draft: str,
    terms: List[Dict[str, Any]],
) -> str:
    """
    Refine a draft translation using a Mistral-compatible chat-completions endpoint.

    This is a best-effort step:
    - If Mistral is not configured, it returns the draft unchanged.
    - If the request fails or the response is malformed, it returns the draft unchanged.

    Args:
        src_lang: Source language code.
        tgt_lang: Target language code.
        src_text: Original source text.
        draft: Draft translation to refine.
        terms: Glossary suggestions extracted from the source text.

    Returns:
        The refined translation if available, otherwise the original draft.
    """
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
        # Intentionally swallow errors to keep the endpoint resilient.
        return draft


# FastAPI app
app = FastAPI(title="Rollomatic Translator API", version="1.0")

# Configure CORS for web clients (UI, browsers, etc.).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else [o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    """Request payload for the /translate endpoint."""
    src_lang: str = Field(..., pattern="^(fr|en|de|it)$")
    tgt_lang: str = Field(..., pattern="^(fr|en|de|it)$")
    text: str = Field(..., min_length=1, max_length=10000)


class TranslateResponse(BaseModel):
    """Response payload for the /translate endpoint."""
    translation: str
    draft: Optional[str] = None
    terms: Optional[List[Dict[str, Any]]] = None


@app.on_event("startup")
def _startup() -> None:
    """
    FastAPI startup hook.

    Loads the glossary into memory to avoid re-reading the CSV per request.

    Raises:
        RuntimeError: If the glossary file does not exist.
    """
    if not os.path.exists(GLOSSARY_PATH):
        raise RuntimeError(f"Glossary file not found: {GLOSSARY_PATH}")
    load_glossary(GLOSSARY_PATH)


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health check endpoint for uptime monitoring.

    Returns:
        A small JSON payload indicating the service is running.
    """
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest, x_access_key: str = Header(default="")) -> TranslateResponse:
    """
    Translate a text from src_lang to tgt_lang, using:
    1) SuperText for a high-quality draft translation,
    2) Optional Mistral refinement with glossary-aware post-editing.

    Security:
        Requires the caller to pass `x-access-key` header matching ACCESS_KEY.

    Args:
        req: Validated translation request body.
        x_access_key: Access key header used to protect the endpoint.

    Returns:
        TranslateResponse containing:
        - translation: final output (refined if Mistral enabled)
        - draft: SuperText output (useful for debugging/comparison)
        - terms: matched glossary terms (useful for UI display)
                 Each term includes ALL candidate translations if multiple exist.

    Raises:
        HTTPException:
            - 500 if server is misconfigured (missing ACCESS_KEY)
            - 401 if the provided key is invalid
            - 500 for any unexpected translation failure
    """
    # Access gate
    if not ACCESS_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured.")
    if x_access_key != ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key.")

    # Avoid unnecessary API calls when source and target are identical.
    if req.src_lang == req.tgt_lang:
        return TranslateResponse(translation=req.text, draft=req.text, terms=[])

    try:
        terms = extract_terms(req.src_lang, req.tgt_lang, req.text)
        draft = supertext_translate(req.src_lang, req.tgt_lang, req.text)
        final = refine_with_mistral(req.src_lang, req.tgt_lang, req.text, draft, terms)

        # Always return translation, draft, and terms
        return TranslateResponse(translation=final, draft=draft, terms=terms)

    except Exception:
        # Keep error surface minimal (avoid leaking provider details to clients).
        raise HTTPException(status_code=500, detail="Translation failed.")

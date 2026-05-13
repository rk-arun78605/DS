"""
claude_vision.py — AI Vision product analysis
Supports: Anthropic Claude claude-sonnet-4-6  |  OpenAI GPT-4o
Multi-item: returns a list of items found in the image.
"""
import base64
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o"

# ── Shared prompt (same for both providers) ───────────────────────────────────
SYSTEM_PROMPT = """You are a precise product identification AI for retail inventory.
Identify EVERY distinct product visible in the image.
Never confuse a single retail unit with a bulk case.
Return ONLY a single valid JSON object. No markdown, no extra text."""

ANALYSIS_PROMPT = """Analyse this product image carefully.

If the image contains MULTIPLE DIFFERENT products (assorted items, mixed basket, shelf with several products),
identify EACH product separately. Return ALL items you can identify.

If the image contains ONE product (or multiple units of the SAME product), return just that one item.

Return this JSON structure:

{
  "scene_description": "one sentence describing the overall image",
  "items": [
    {
      "product_name":        "specific name — e.g. 'Fresh Dill', 'Anchor Whipped Cream 250ml'",
      "brand":               "exact brand from label or null",
      "category":            "FNV | Food | Beverage | Alcohol | Electronics | Household | Personal_Care | Clothing | Other",
      "product_type":        "specific type — e.g. 'Fresh Herb', 'UHT Cream', 'Soft Drink Can'",
      "description":         "one precise sentence describing this specific item",
      "is_fnv":              <true if fresh produce/vegetable/herb, false otherwise>,

      "quantity":            <count of this item's primary sellable units>,
      "unit_of_measurement": "pcs | case | kg | g | ml | L",
      "units_per_case":      <integer or null>,
      "num_cases":           <integer>,
      "num_pieces":          <integer>,
      "total_units":         <num_cases * units_per_case + num_pieces>,

      "grammage":            "size/weight on pack — e.g. '250ml', '500g', '6x330ml' — null for FNV",

      "weight_per_unit":     <numeric weight from scale or label; null if not visible>,
      "weight_unit":         "kg | g | lb | oz | null",
      "total_weight":        <weight_per_unit * total_units or null>,
      "scale_reading":       "exact scale display text or null",

      "expiry_date":         "BB / EXP / USE BY date as printed — null for FNV or if not visible",
      "has_expiry":          <true if expiry visible, false if packaged but no date, null if FNV>,

      "is_alcohol":          <true | false>,
      "alcohol_type":        "Whisky | Vodka | Gin | Rum | Wine | Beer | Brandy | Tequila | Champagne | Liqueur | Other | null",
      "alcohol_subtype":     "e.g. Single Malt Scotch — null if not alcohol",
      "bottle_size_ml":      <capacity ml; null if not alcohol>,
      "fill_level_pct":      <0-100 remaining %; null if sealed or not alcohol>,
      "ml_remaining":        <bottle_size_ml * fill_level_pct / 100; null if not applicable>,
      "ml_consumed":         <bottle_size_ml - ml_remaining; null if not applicable>,
      "alcohol_percentage":  <ABV% from label; null if not visible>,

      "barcode":             "digits if visible; null if not",
      "country_of_origin":   "country if visible; null if not",

      "search_keywords":     ["brand", "product noun", "grammage if known"],

      "confidence":          <0.00-1.00 for this specific item>,
      "notes":               "reasoning or uncertainty for this item"
    }
  ]
}

═══ CRITICAL IDENTIFICATION RULES ═══

PACKAGING (single vs case):
  Single retail unit → UOM="pcs", num_pieces=1
  One tetra pack / can / bottle / jar / box → pcs, NOT cartons
  Multiple units bundled → UOM="case", num_cases=X

FNV herbs (identify by shape, not guess):
  Dill      = feathery yellow-green fronds, thin stems
  Parsley   = dark curly or flat broad leaves
  Coriander = round serrated bright-green leaves
  Mint      = oval serrated leaves
  Cucumber  = long dark-green cylinder
  NEVER use "/" between two different items

BRAND: null if not visible on label — never write "Unknown" or "Unbranded"

EXPIRY: for non-FNV, read BEST BEFORE / BB / EXP / USE BY text exactly as printed

SCALE: if digital scale display visible, read the exact digits shown

Return ONLY the JSON. No markdown fences. No extra text."""


# ── Key helpers ───────────────────────────────────────────────────────────────

def _get_provider() -> str:
    try:
        from django.conf import settings
        return getattr(settings, "AI_VISION_PROVIDER", "anthropic").lower()
    except Exception:
        return os.environ.get("AI_VISION_PROVIDER", "anthropic").lower()


def _get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            from django.conf import settings
            key = getattr(settings, "ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return key


def _get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            from django.conf import settings
            key = getattr(settings, "OPENAI_API_KEY", "")
        except Exception:
            pass
    return key


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"items": [{"product_name": "Parse error", "notes": raw[:500], "confidence": 0.0}]}


# ── Anthropic Claude ──────────────────────────────────────────────────────────

def _call_anthropic(image_bytes: bytes, mime_type: str) -> dict:
    import anthropic
    key = _get_anthropic_key()
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set. Set it via system env var or Django settings.")

    client = anthropic.Anthropic(api_key=key)
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                {"type": "text",  "text": ANALYSIS_PROMPT},
            ],
        }],
    )
    raw = msg.content[0].text
    result = _parse_json(raw)
    result["_model"]    = CLAUDE_MODEL
    result["_provider"] = "anthropic"
    return result


# ── OpenAI GPT-4o ─────────────────────────────────────────────────────────────

def _call_openai(image_bytes: bytes, mime_type: str) -> dict:
    import openai as _openai
    key = _get_openai_key()
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Set it via system env var or Django settings.")

    client = _openai.OpenAI(api_key=key)
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": SYSTEM_PROMPT + "\n\n" + ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ],
        }],
    )
    raw = resp.choices[0].message.content
    result = _parse_json(raw)
    result["_model"]    = OPENAI_MODEL
    result["_provider"] = "openai"
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Analyse image with configured provider.
    Returns dict with 'items' list — one entry per identified product.
    Falls back to the other provider automatically if one fails.
    """
    provider = _get_provider()

    primary   = _call_anthropic if provider == "anthropic" else _call_openai
    secondary = _call_openai    if provider == "anthropic" else _call_anthropic

    try:
        result = primary(image_bytes, mime_type)
    except Exception as e_primary:
        logger.warning("Primary provider (%s) failed: %s — trying fallback", provider, e_primary)
        try:
            result = secondary(image_bytes, mime_type)
            result["_fallback"] = str(e_primary)
        except Exception as e_secondary:
            raise ValueError(
                f"Both AI providers failed.\n"
                f"Primary ({provider}): {e_primary}\n"
                f"Secondary: {e_secondary}"
            )

    # Normalise: ensure result always has an 'items' list
    if "items" not in result:
        # Old single-item format — wrap it
        item = {k: v for k, v in result.items() if not k.startswith("_")}
        result["items"] = [item]

    # Ensure each item has at least the required keys with defaults
    for item in result.get("items", []):
        item.setdefault("product_name", None)
        item.setdefault("confidence",   0.0)
        item.setdefault("is_alcohol",   False)
        item.setdefault("is_fnv",       False)
        item.setdefault("has_expiry",   None)

    return result

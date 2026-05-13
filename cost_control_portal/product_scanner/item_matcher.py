"""
item_matcher.py
===============
Ports the ProductMatcher logic (TF-IDF + WRatio + grammage/discriminating penalties)
to match AI-identified product descriptions against cc_item_master.

Index is built once and cached in-process (module-level). It is rebuilt
automatically whenever cc_item_master row count changes.
"""

import re
import threading

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Tuning (same as ProductMatcher) ───────────────────────────────────────────
PRE_FILTER            = 50
GRAM_MISMATCH_PENALTY = 50
GRAM_ABSENT_PENALTY   = 10
DISCRIM_GROUP_PENALTY     = 45
DISCRIM_SINGLETON_PENALTY = 25
DISCRIM_SINGLETON_CAP     = 50

_DISCRIM_GROUPS: list[frozenset] = [
    frozenset({'gravy', 'jelly', 'brine', 'pate', 'broth', 'sauce', 'water', 'juice'}),
    frozenset({'puppy', 'kitten', 'senior', 'adult', 'junior'}),
    frozenset({'decaf', 'regular'}),
]
_DISCRIM_SINGLETONS: frozenset = frozenset({
    'fish', 'salmon', 'tuna', 'cod', 'chicken', 'beef', 'lamb', 'pork',
    'turkey', 'duck', 'rabbit', 'liver', 'kidney',
    'organic', 'mint', 'lemon', 'strawberry', 'vanilla', 'chocolate',
    'original', 'sensitive', 'whitening', 'antibacterial',
})

ABBREV_MAP = {
    r'\bltr\b': 'litre', r'\bltrs\b': 'litres',
    r'\bpk\b':  'pack',  r'\bpkt\b':  'packet',
    r'\bpcs\b': 'pieces',r'\bbtl\b':  'bottle',
    r'\bbtls\b':'bottles',r'\bkg\b':  'kg',
    r'\bgm\b':  'gram',  r'\bgms\b':  'grams',
    r'\bchoc\b':'chocolate',r'\bveg\b':'vegetable',
}

_GRAM_RAW_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(ml|cl|l\b|ltr[s]?|litre[s]?|kg|g\b|gm[s]?|gram[s]?|mg|oz)',
    re.IGNORECASE,
)
_PACK_RAW_RE = re.compile(
    r'(\d+)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*(ml|cl|l\b|ltr[s]?|kg|g\b|gm[s]?)',
    re.IGNORECASE,
)
_COUNT_RAW_RE = re.compile(
    r'(\d+)\s*(?:pk\b|pack[s]?\b|pcs\b|pieces?\b|sachets?\b|cans?\b|bottles?\b)',
    re.IGNORECASE,
)
_UNIT_NORM = {
    'l':'ltr','ltr':'ltr','ltrs':'ltr','litre':'ltr','litres':'ltr',
    'gm':'g','gms':'g','gram':'g','grams':'g','ml':'ml','cl':'cl',
    'kg':'kg','oz':'oz','g':'g',
}


def _normalize(text: str) -> str:
    if not text:
        return ''
    text = str(text).lower().strip()
    for pat, rep in ABBREV_MAP.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _gram_tokens(text: str) -> frozenset:
    tokens: set = set()
    for m in _PACK_RAW_RE.finditer(str(text)):
        u = _UNIT_NORM.get(m.group(3).lower(), m.group(3).lower())
        tokens.add(f"{m.group(2)}{u}"); tokens.add(f"{m.group(1)}pk")
    for m in _GRAM_RAW_RE.finditer(str(text)):
        u = _UNIT_NORM.get(m.group(2).lower(), m.group(2).lower())
        tokens.add(f"{m.group(1)}{u}")
    for m in _COUNT_RAW_RE.finditer(str(text)):
        tokens.add(f"{m.group(1)}pk")
    return frozenset(tokens)


def _discrim_tokens(text: str) -> frozenset:
    words = set(_normalize(str(text)).split())
    all_d = _DISCRIM_SINGLETONS | {w for g in _DISCRIM_GROUPS for w in g}
    return frozenset(words & all_d)


# ── In-process index cache ─────────────────────────────────────────────────────
_lock        = threading.Lock()
_cache: dict = {}   # keys: 'df', 'norm', 'gram', 'discrim', 'vec', 'mat', 'n_rows'


def _load_index() -> dict:
    """Load cc_item_master from DB and build TF-IDF index. Cached in-process."""
    global _cache
    from django.db import connection

    with connection.cursor() as cur:
        cur.execute(
            "SELECT item_code, item_name, department, grp, sub_group, uom "
            "FROM cc_item_master WHERE is_active = true ORDER BY item_name"
        )
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()

    n = len(rows)
    with _lock:
        if _cache.get('n_rows') == n and _cache.get('df') is not None:
            return _cache

        df = pd.DataFrame(rows, columns=cols)
        norm   = [_normalize(x) for x in df['item_name']]
        gram   = [_gram_tokens(x)    for x in df['item_name']]
        discrim= [_discrim_tokens(x) for x in df['item_name']]
        vec    = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3),
                                  min_df=1, max_features=40_000)
        mat    = vec.fit_transform(norm)
        _cache = {'df': df, 'norm': norm, 'gram': gram,
                  'discrim': discrim, 'vec': vec, 'mat': mat, 'n_rows': n}
    return _cache


def match_product(product_name: str, brand: str = "",
                  keywords: list | None = None) -> list[dict]:
    """
    Match AI-identified product against cc_item_master.
    Returns up to 5 matches: [{item_code, item_name, department, grp, sub_group, uom, score, confidence}]
    """
    # Build query string: brand + product_name + keywords (same as supplier description)
    parts = [x.strip() for x in [brand, product_name] + (keywords or []) if x and x.strip()]
    query_raw  = " ".join(parts)
    query_norm = _normalize(query_raw)
    if not query_norm:
        return []

    try:
        idx = _load_index()
    except Exception as e:
        return []

    df     = idx['df']
    norm   = idx['norm']
    gram   = idx['gram']
    discrim= idx['discrim']
    vec    = idx['vec']
    mat    = idx['mat']

    if df.empty:
        return []

    # TF-IDF pre-filter → top PRE_FILTER candidates
    q_vec   = vec.transform([query_norm])
    sims    = cosine_similarity(q_vec, mat)[0]
    n_cands = min(PRE_FILTER, len(norm))
    top_idx = np.argpartition(sims, -n_cands)[-n_cands:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    cand_names = [norm[k] for k in top_idx]

    q_gram   = _gram_tokens(query_raw)
    q_discrim= _discrim_tokens(query_raw)
    q_brand  = query_norm.split()[0] if query_norm.strip() else ''

    cand_gram   = [gram[int(k)]    for k in top_idx]
    cand_discrim= [discrim[int(k)] for k in top_idx]

    def _scorer(q, c, **kwargs):
        score = fuzz.WRatio(q, c)

        # Brand boost
        c_words = c.split()
        if q_brand and c_words and c_words[0] == q_brand:
            score = min(100, score + 20)

        try:
            ci       = cand_names.index(c)
            c_gram   = cand_gram[ci]
            c_discrim= cand_discrim[ci]
        except ValueError:
            c_gram   = _gram_tokens(c)
            c_discrim= _discrim_tokens(c)

        # Grammage penalty
        if q_gram:
            if c_gram:
                if q_gram.isdisjoint(c_gram):
                    score = max(0, score - GRAM_MISMATCH_PENALTY)
            else:
                score = max(0, score - GRAM_ABSENT_PENALTY)

        # Discriminating group penalty
        for grp in _DISCRIM_GROUPS:
            q_in = q_discrim & grp
            c_in = c_discrim & grp
            if q_in and c_in and q_in.isdisjoint(c_in):
                score = max(0, score - DISCRIM_GROUP_PENALTY)

        # Singleton penalty
        pen = sum(
            DISCRIM_SINGLETON_PENALTY
            for w in _DISCRIM_SINGLETONS
            if (w in q_discrim) != (w in c_discrim)
        )
        if pen:
            score = max(0, score - min(pen, DISCRIM_SINGLETON_CAP))

        return score

    hits = process.extract(query_norm, cand_names, scorer=_scorer, limit=20)

    results = []
    for hit_name, hit_score, hit_rank in hits:
        im_idx = int(top_idx[hit_rank])
        row    = df.iloc[im_idx]
        pct    = round(hit_score, 1)
        conf   = "HIGH" if pct >= 85 else "MEDIUM" if pct >= 65 else "LOW"
        results.append({
            "item_code":   str(row["item_code"]),
            "item_name":   str(row["item_name"]),
            "department":  str(row.get("department") or ""),
            "grp":         str(row.get("grp") or ""),
            "sub_group":   str(row.get("sub_group") or ""),
            "uom":         str(row.get("uom") or ""),
            "match_pct":   pct,
            "confidence":  conf,
        })
    return results

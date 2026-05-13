import re
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_CACHE   = Path(__file__).parent / "_index_cache.pkl"
CACHE_VERSION = 5     # bumped: added discrim_tokens to cache
PRE_FILTER    = 50    # TF-IDF candidates before running WRatio
FUZZY_CHUNK   = 200   # rows per cosine_similarity batch (memory guard)
TOP_N         = 5     # how many alternative matches to return (Top Match 2..N)

# Grammage penalty tuning
GRAM_MISMATCH_PENALTY = 50   # size token present in both but different (500ml vs 250ml)
GRAM_ABSENT_PENALTY   = 10   # candidate has no size but supplier does (uncertain)

# Discriminating keyword penalty tuning
DISCRIM_GROUP_PENALTY     = 45   # wrong format group (GRAVY vs JELLY)
DISCRIM_SINGLETON_PENALTY = 25   # key ingredient present in one but not the other
DISCRIM_SINGLETON_CAP     = 50   # max total singleton penalty per candidate

# ── Mutually-exclusive format/variant groups ───────────────────────────────────
# If the supplier has one word from a group and the candidate has a DIFFERENT
# word from the same group → they are different product variants.
_DISCRIM_GROUPS: list[frozenset] = [
    # Sauce / liquid medium — these make a product fundamentally different
    frozenset({'gravy', 'jelly', 'brine', 'pate', 'broth', 'terrine',
               'sauce', 'water', 'juice'}),
    # Pet food life stage
    frozenset({'puppy', 'kitten', 'senior', 'adult', 'junior'}),
    # Caffeine type (beverages)
    frozenset({'decaf', 'regular'}),
]

# ── Key ingredient / descriptor words (singletons) ────────────────────────────
# If clearly present in candidate but absent from supplier (or vice versa),
# apply a penalty — these strongly distinguish product variants.
_DISCRIM_SINGLETONS: frozenset = frozenset({
    # Proteins — the most common source of wrong matches in food/pet food
    'fish', 'salmon', 'tuna', 'trout', 'cod', 'mackerel', 'sardine',
    'chicken', 'beef', 'lamb', 'pork', 'turkey', 'duck', 'rabbit',
    'venison', 'liver', 'kidney',
    # Key food descriptors
    'organic', 'mint', 'lemon', 'strawberry', 'vanilla', 'chocolate',
    'original', 'sensitive', 'whitening', 'antibacterial',
})

ABBREV_MAP = {
    r'\bltr\b':    'litre',  r'\bltrs\b':   'litres',
    r'\bpk\b':     'pack',   r'\bpkt\b':    'packet',
    r'\bpcs\b':    'pieces', r'\bpc\b':     'piece',
    r'\bbtl\b':    'bottle', r'\bbtls\b':   'bottles',
    r'\bkg\b':     'kg',     r'\bgm\b':     'gram',
    r'\bgms\b':    'grams',  r'\bno\b':     'number',
    r'\bnos\b':    'numbers',r'\bqty\b':    'quantity',
    r'\basstd\b':  'assorted',r'\bveg\b':   'vegetable',
    r'\bchoc\b':   'chocolate',r'\bvanl\b': 'vanilla',
    r'\bstrawb\b': 'strawberry',
}

_WEIGHT_PKG_RE = re.compile(
    r'\d+(?:\.\d+)?\s*(?:ml|ltr|ltrs|litre|litres|kg|gm|gms|gram|grams|g\b)|'
    r'pk\s*\d+|\d+\s*pk\b|'
    r'\d+\s*x\s*\d+|'
    r'\d+\s*(?:pack|packs|pcs|pieces|piece|bottle|bottles|can|cans|sachet|sachets)',
    re.IGNORECASE,
)


# ── Grammage extraction ────────────────────────────────────────────────────────
# Runs on RAW (un-normalised) text so that decimals like 1.5KG are preserved.

_GRAM_RAW_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(ml|cl|l\b|ltr[s]?|litre[s]?|kg|g\b|gm[s]?|gram[s]?|mg|oz|fl\.?\s*oz)',
    re.IGNORECASE,
)
_PACK_RAW_RE = re.compile(
    r'(\d+)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*'
    r'(ml|cl|l\b|ltr[s]?|litre[s]?|kg|g\b|gm[s]?|gram[s]?)',
    re.IGNORECASE,
)
_COUNT_RAW_RE = re.compile(
    r'(\d+)\s*(?:pk\b|pck\b|pack[s]?\b|pcs\b|pieces?\b|sachets?\b|cans?\b|bottles?\b|tins?\b)',
    re.IGNORECASE,
)

_GRAM_UNIT_NORM = {
    'l': 'ltr', 'ltr': 'ltr', 'ltrs': 'ltr', 'litre': 'ltr', 'litres': 'ltr',
    'gm': 'g', 'gms': 'g', 'gram': 'g', 'grams': 'g', 'mg': 'mg',
    'ml': 'ml', 'cl': 'cl', 'kg': 'kg', 'oz': 'oz', 'g': 'g',
    'fl oz': 'floz', 'fl. oz': 'floz',
}


def _extract_gram_tokens(text: str) -> frozenset:
    """
    Extract normalised size/weight/count tokens from raw (un-normalised) text.

    Examples:
      "Pepsi Cola 500ML"              → frozenset({'500ml'})
      "Dettol Soap 100G x 4 Pack"    → frozenset({'100g', '4pk'})
      "Sunlight 1.5KG"               → frozenset({'1.5kg'})
      "Rice 25 KG"                   → frozenset({'25kg'})
      "Coca Cola 330ML 24 Pack"       → frozenset({'330ml', '24pk'})
      "2 x 500ML Pepsi"              → frozenset({'500ml', '2pk'})
    """
    if not text:
        return frozenset()
    text = str(text)
    tokens: set[str] = set()

    # "2 x 500ml" — extract the unit size (500ml) and the multiplier as count (2pk)
    for m in _PACK_RAW_RE.finditer(text):
        count = m.group(1)
        qty   = m.group(2)
        unit  = _GRAM_UNIT_NORM.get(m.group(3).lower().replace(' ', ''), m.group(3).lower())
        tokens.add(f"{qty}{unit}")
        tokens.add(f"{count}pk")

    # Plain weight/volume: 500ml, 1.5kg, 100g …
    for m in _GRAM_RAW_RE.finditer(text):
        qty  = m.group(1)
        raw_unit = m.group(2).lower().replace(' ', '').rstrip('.')
        unit = _GRAM_UNIT_NORM.get(raw_unit, raw_unit)
        tokens.add(f"{qty}{unit}")

    # Pack/count: 24pk, 24 pack, 6 cans …
    for m in _COUNT_RAW_RE.finditer(text):
        tokens.add(f"{m.group(1)}pk")

    return frozenset(tokens)


def _extract_discrim_tokens(text: str) -> frozenset:
    """
    Extract discriminating variant tokens from normalized text.
    Returns the subset of _DISCRIM_GROUPS words + _DISCRIM_SINGLETONS
    that actually appear in the text.
    """
    if not text:
        return frozenset()
    norm = _normalize(str(text))
    words = set(norm.split())
    all_discrim = _DISCRIM_SINGLETONS | {w for g in _DISCRIM_GROUPS for w in g}
    return frozenset(words & all_discrim)


# ── Utilities ──────────────────────────────────────────────────────────────────
def extract_weight_pkg(text: str) -> str:
    if not text:
        return ''
    found = _WEIGHT_PKG_RE.findall(str(text))
    seen, out = set(), []
    for f in found:
        key = f.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(f.strip())
    return ' · '.join(out)


def _clean_barcode(val) -> str:
    if val is None:
        return ''
    s = str(val).strip()
    if s.endswith('.0'):
        base = s[:-2]
        if base.lstrip('-').isdigit():
            s = base
    return s


def _normalize(text: str) -> str:
    if not text:
        return ''
    text = str(text).lower().strip()
    for pattern, replacement in ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _confidence(score: float) -> str:
    if score > 0.85:
        return 'HIGH'
    if score >= 0.65:
        return 'MEDIUM'
    return 'LOW'


def _fingerprint(df: pd.DataFrame) -> str:
    """Fast fingerprint: row count + hash of first/last 50 rows."""
    h = hashlib.md5(str(len(df)).encode())
    sample = pd.concat([df.head(50), df.tail(50)]).to_csv(index=False).encode()
    h.update(sample)
    return h.hexdigest()


# ── Index build (with disk cache) ─────────────────────────────────────────────
def build_index(itemmaster_df: pd.DataFrame):
    """
    Returns (im, norm_names, bc_dict, tfidf_vec, tfidf_mat).
    Saves to disk on first build; loads instantly on subsequent sessions.
    Cache is invalidated automatically when itemmaster changes.
    """
    fp = _fingerprint(itemmaster_df)

    # Try loading from disk
    if INDEX_CACHE.exists():
        try:
            with open(INDEX_CACHE, 'rb') as f:
                cached = pickle.load(f)
            if (cached.get('version') == CACHE_VERSION
                    and cached.get('fingerprint') == fp):
                return (cached['im'], cached['norm_names'],
                        cached['gram_tokens'], cached['discrim_tokens'],
                        cached['bc_dict'], cached['tfidf_vec'], cached['tfidf_mat'])
        except Exception:
            pass  # corrupted cache — rebuild

    # ── Build fresh ────────────────────────────────────────────────────────────
    im = itemmaster_df.reset_index(drop=True)

    # barcode → list of row indices (captures ALL items sharing a barcode)
    bc_dict: dict[str, list[int]] = {}
    for idx, raw_bc in enumerate(im['barcode']):
        bc = _clean_barcode(raw_bc)
        if bc:
            bc_dict.setdefault(bc, []).append(idx)

    # Pre-normalize all item names (done once, reused every run)
    norm_names = [_normalize(n) for n in im['item_name']]

    # Pre-extract grammage tokens from raw item names (done once, used in scoring)
    gram_tokens: list[frozenset] = [
        _extract_gram_tokens(n) for n in im['item_name']
    ]

    # Pre-extract discriminating variant tokens (done once, used in scoring)
    discrim_tokens: list[frozenset] = [
        _extract_discrim_tokens(n) for n in im['item_name']
    ]

    # TF-IDF index for fast pre-filtering (reduces 300k comparisons → 50)
    tfidf_vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 3),
        min_df=1,
        max_features=60_000,   # cap vocabulary size to keep matrix small
    )
    tfidf_mat = tfidf_vec.fit_transform(norm_names)   # sparse matrix

    # Save to disk
    try:
        with open(INDEX_CACHE, 'wb') as f:
            pickle.dump({
                'version':        CACHE_VERSION,
                'fingerprint':    fp,
                'im':             im,
                'norm_names':     norm_names,
                'gram_tokens':    gram_tokens,
                'discrim_tokens': discrim_tokens,
                'bc_dict':        bc_dict,
                'tfidf_vec':      tfidf_vec,
                'tfidf_mat':      tfidf_mat,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # non-fatal if disk write fails

    return im, norm_names, gram_tokens, discrim_tokens, bc_dict, tfidf_vec, tfidf_mat


# ── Main matching ──────────────────────────────────────────────────────────────
def run_matching(
    uploaded_df: pd.DataFrame,
    im: pd.DataFrame,
    norm_names: list[str],
    gram_tokens: list[frozenset],
    discrim_tokens: list[frozenset],
    bc_dict: dict[str, int],
    tfidf_vec: TfidfVectorizer,
    tfidf_mat,
    progress_cb=None,
) -> pd.DataFrame:
    total   = len(uploaded_df)
    results: list[dict | None] = [None] * total
    fuzzy_queue: list[tuple[int, dict, str]] = []

    # ── Phase 1: exact barcode — O(1) dict lookup ─────────────────────────────
    for i, (_, row) in enumerate(uploaded_df.iterrows()):
        bc   = _clean_barcode(row.get('barcode', ''))
        bc1  = _clean_barcode(row.get('barcode1', ''))
        desc = str(row.get('item_description', '') or '').strip()

        base = {
            'Supplier Description': desc,
            'Weight / Pack':        extract_weight_pkg(desc),
            'Input Barcode':        bc,
            'Input Barcode1':       bc1,
            'Match Type':           'NO_MATCH',
            'Match %':              '',
            'Matched Item Name':    '',
            'Item Code':            '',
            'Matched Barcode':      '',
            'Confidence':           '',
            'Duplicate Item Codes': '',
            'Remark':               '',
            **{f'Top Match {n}': '' for n in range(2, TOP_N + 1)},
        }

        matched = False
        for test_bc, mtype in [(bc, 'EXACT'), (bc1, 'EXACT_SECONDARY')]:
            if test_bc and test_bc in bc_dict:
                idxs  = bc_dict[test_bc]
                first = im.iloc[idxs[0]]
                base.update({
                    'Match Type':        mtype,
                    'Match %':           100.0,
                    'Matched Item Name': first['item_name'],
                    'Item Code':         first['item_code'],
                    'Matched Barcode':   first['barcode'],
                    'Confidence':        'HIGH',
                })
                if len(idxs) > 1:
                    dup_codes = ', '.join(im.iloc[k]['item_code'] for k in idxs)
                    dup_items = ' | '.join(
                        f"{im.iloc[k]['item_code']} — {im.iloc[k]['item_name']}"
                        for k in idxs
                    )
                    base['Duplicate Item Codes'] = dup_codes
                    base['Remark'] = f'DUPLICATE BARCODE ({len(idxs)} items): {dup_items}'
                matched = True
                break

        if matched:
            results[i] = base
        elif desc:
            fuzzy_queue.append((i, base, _normalize(desc)))
        else:
            results[i] = base

    exact_count = total - len(fuzzy_queue)
    if progress_cb:
        progress_cb(0.4, f"Exact matches done ({exact_count:,} matched, "
                         f"{len(fuzzy_queue):,} need fuzzy)…")

    # ── Phase 2: TF-IDF pre-filter → brand-aware WRatio on top-50 candidates ───
    if fuzzy_queue:
        nq = len(fuzzy_queue)

        for ci in range(0, nq, FUZZY_CHUNK):
            chunk      = fuzzy_queue[ci : ci + FUZZY_CHUNK]
            q_texts    = [q[2] for q in chunk]
            q_vecs     = tfidf_vec.transform(q_texts)
            sim_scores = cosine_similarity(q_vecs, tfidf_mat)

            for j, (orig_idx, base, norm_desc) in enumerate(chunk):
                row_sims  = sim_scores[j]
                top_cands = np.argpartition(row_sims, -PRE_FILTER)[-PRE_FILTER:]
                top_cands = top_cands[np.argsort(row_sims[top_cands])[::-1]]
                cand_names = [norm_names[k] for k in top_cands]

                # Pre-compute per-query tokens (done once per supplier row)
                raw_desc   = base['Supplier Description']
                q_gram     = _extract_gram_tokens(raw_desc)
                q_discrim  = _extract_discrim_tokens(raw_desc)
                q_brand    = norm_desc.split()[0] if norm_desc.strip() else ''

                # Pre-fetch candidate tokens by index (O(1) list lookups)
                cand_gram    = [gram_tokens[int(k)]    for k in top_cands]
                cand_discrim = [discrim_tokens[int(k)] for k in top_cands]

                def _scorer(q, c, **kwargs):
                    score = fuzz.WRatio(q, c)

                    # ── Brand boost ───────────────────────────────────────────
                    c_words = c.split()
                    if q_brand and c_words and c_words[0] == q_brand:
                        score = min(100, score + 20)

                    # Look up pre-computed tokens for this candidate
                    try:
                        ci       = cand_names.index(c)
                        c_gram   = cand_gram[ci]
                        c_discrim = cand_discrim[ci]
                    except ValueError:
                        c_gram   = _extract_gram_tokens(c)
                        c_discrim = _extract_discrim_tokens(c)

                    # ── Grammage penalty (500ml vs 250ml) ─────────────────────
                    if q_gram:
                        if c_gram:
                            if q_gram.isdisjoint(c_gram):
                                score = max(0, score - GRAM_MISMATCH_PENALTY)
                        else:
                            score = max(0, score - GRAM_ABSENT_PENALTY)

                    # ── Discriminating format group penalty ───────────────────
                    # e.g. supplier says GRAVY, candidate says JELLY → wrong product
                    for group in _DISCRIM_GROUPS:
                        q_in = q_discrim & group
                        c_in = c_discrim & group
                        if q_in and c_in and q_in.isdisjoint(c_in):
                            score = max(0, score - DISCRIM_GROUP_PENALTY)

                    # ── Singleton ingredient / descriptor penalty ─────────────
                    # e.g. candidate has FISH but supplier has no FISH
                    # e.g. supplier has CHICKEN but candidate has none
                    singleton_penalty = 0
                    for word in _DISCRIM_SINGLETONS:
                        q_has = word in q_discrim
                        c_has = word in c_discrim
                        if q_has != c_has:
                            singleton_penalty += DISCRIM_SINGLETON_PENALTY
                    if singleton_penalty:
                        score = max(0, score - min(singleton_penalty,
                                                   DISCRIM_SINGLETON_CAP))

                    return score

                hits = process.extract(norm_desc, cand_names, scorer=_scorer, limit=TOP_N)

                if hits:
                    best_score  = hits[0][1] / 100.0
                    best_im_idx = int(top_cands[hits[0][2]])
                    best        = im.iloc[best_im_idx]
                    base.update({
                        'Match Type':        'FUZZY',
                        'Match %':           round(best_score * 100, 1),
                        'Matched Item Name': best['item_name'],
                        'Item Code':         best['item_code'],
                        'Matched Barcode':   best['barcode'],
                        'Confidence':        _confidence(best_score),
                    })
                    # Top Match 2..TOP_N — each includes item code, name, barcode
                    for rank, hit in enumerate(hits[1:], start=2):
                        alt    = im.iloc[int(top_cands[hit[2]])]
                        score  = round(hit[1], 1)
                        base[f'Top Match {rank}'] = (
                            f"{alt['item_code']} — {alt['item_name']} "
                            f"({alt['barcode']}) [{score}%]"
                        )

                results[orig_idx] = base

            if progress_cb:
                done = min(ci + FUZZY_CHUNK, nq)
                progress_cb(0.4 + 0.55 * done / nq,
                            f"Fuzzy matching: {done:,} / {nq:,} rows…")

    if progress_cb:
        progress_cb(1.0, "Done")

    return pd.DataFrame(results)

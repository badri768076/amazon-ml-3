"""Metrics, EDA, dataset quality report and error analysis.

Challenge metric: **macro F0.5 over Source-1 entities**. For each entity with true
match set T and predicted set P:

    T = {} and P = {}      -> 1   (correct no-match / singleton)
    T = {} xor P = {}      -> 0
    otherwise              -> F0.5(precision = |P∩T|/|P|, recall = |P∩T|/|T|)

    F0.5 = 1.25 * P * R / (0.25 * P + R),   macro F0.5 = mean over all entities

Micro (pooled over matched IDs) precision / recall / F0.5 / F1 are reported as
well. True matches that blocking never retrieved count as missed matches, so
validation scores are honest end to end.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import duckdb
import numpy as np
import polars as pl

from config import ALL_TABLES, SOURCE_COLUMNS, Config

log = logging.getLogger("ber.eval")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def f_beta(p, r, beta2: float = 0.25):
    return np.where((p + r) > 0, (1 + beta2) * p * r / np.maximum(beta2 * p + r, 1e-12), 0.0)


def prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f0_5": f05, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


class GroupEvaluator:
    """Vectorised entity-level evaluation of many decision rules on one scored pair set.

    ``s1_keys`` (sorted) groups the candidate pairs by Source-1 entity; ``gt_counts``
    maps *every* entity in scope (including ones without candidates) to its number
    of true matches.
    """

    def __init__(self, s1_keys: np.ndarray, labels: np.ndarray, s1_ids: np.ndarray,
                 gt_counts: dict, objective: str = "macro_f0_5"):
        self.labels = labels.astype(bool)
        self.objective = objective
        _, self.starts, self.group_of = np.unique(s1_keys, return_index=True, return_inverse=True)
        first_ids = s1_ids[self.starts]
        self.n_true = np.array([gt_counts.get(x, 0) for x in first_ids], dtype=np.float64)
        in_groups = set(first_ids.tolist())
        rest = [c for e, c in gt_counts.items() if e not in in_groups]
        self.n_entities = len(gt_counts)
        self.nc_zero = sum(1 for c in rest if c == 0)          # no candidates, no truth -> correct
        self.nc_pos = sum(1 for c in rest if c > 0)            # no candidates, has truth -> missed
        self.n_true_total = int(sum(gt_counts.values()))
        counts = np.array(list(gt_counts.values()))
        self.n_zero_entities = int((counts == 0).sum())
        self.n_one_entities = int((counts == 1).sum())
        self.n_multi_entities = int((counts > 1).sum())

    def group_max(self, x: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(x, self.starts)[self.group_of]

    def group_rank(self, x: np.ndarray) -> np.ndarray:
        """1-based descending rank of x inside its group."""
        order = np.lexsort((-x, self.group_of))
        ranks = np.empty(len(x), dtype=np.int32)
        ranks[order] = np.arange(len(x)) - self.starts[self.group_of[order]] + 1
        return ranks

    def evaluate(self, pred: np.ndarray) -> dict:
        pred = pred.astype(bool)
        if len(pred):
            tp_g = np.add.reduceat((pred & self.labels).astype(np.int64), self.starts).astype(np.float64)
            np_g = np.add.reduceat(pred.astype(np.int64), self.starts).astype(np.float64)
        else:
            tp_g = np_g = np.zeros(0)
        nt = self.n_true
        prec = np.divide(tp_g, np_g, out=np.zeros_like(tp_g), where=np_g > 0)
        rec = np.divide(tp_g, nt, out=np.zeros_like(tp_g), where=nt > 0)
        f_ent = np.where(nt == 0, (np_g == 0).astype(np.float64), f_beta(prec, rec))
        macro = (f_ent.sum() + self.nc_zero) / max(1, self.n_entities)
        tp, npred = float(tp_g.sum()), float(np_g.sum())
        m = prf(tp, npred - tp, self.n_true_total - tp)
        m = {"micro_" + k if k in ("precision", "recall", "f0_5", "f1") else k: v for k, v in m.items()}
        zero_ok = int(((nt == 0) & (np_g == 0)).sum()) + self.nc_zero
        one_ok = int(((nt == 1) & (tp_g == 1) & (np_g == 1)).sum())
        multi_ok = int(((nt > 1) & (tp_g == nt) & (np_g == nt)).sum())
        m.update({
            "macro_f0_5": float(macro),
            # entity-averaged precision / recall over entities where they are defined
            "macro_precision": float(prec[np_g > 0].mean()) if (np_g > 0).any() else 0.0,
            "macro_recall": float(rec[nt > 0].sum() / max(1, (nt > 0).sum() + self.nc_pos)),
            "false_merges": m["fp"], "missed_matches": m["fn"],
            "zero_match_correct": zero_ok, "zero_match_total": self.n_zero_entities,
            "singleton_correctness": zero_ok / self.n_zero_entities if self.n_zero_entities else None,
            "one_match_exact": one_ok, "one_match_total": self.n_one_entities,
            "multi_match_exact": multi_ok, "multi_match_total": self.n_multi_entities,
            "avg_matches_per_s1": npred / max(1, self.n_entities),
            "s1_with_no_prediction": int(self.n_entities - (np_g > 0).sum()),
        })
        # aliases used in logs / reports
        m["precision"], m["recall"], m["f0_5"] = m["micro_precision"], m["micro_recall"], m["micro_f0_5"]
        m["objective"] = m[self.objective]
        return m


def entity_f05_from_sets(pred: dict[str, set], truth: dict[str, set]) -> float:
    """Reference (slow, set-based) macro F0.5 used to cross-check the vectorised evaluator."""
    scores = []
    for e, t in truth.items():
        p = pred.get(e, set())
        if not t and not p:
            scores.append(1.0)
        elif not t or not p:
            scores.append(0.0)
        else:
            tp = len(p & t)
            pr, rc = tp / len(p), tp / len(t)
            scores.append(1.25 * pr * rc / (0.25 * pr + rc) if tp else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
NULL_LIKE = ["na", "n/a", "null", "none", "nan", "-", "--", "?"]
SOURCE_TABLES = [t for t in ALL_TABLES if not t.endswith("ground_truth")]


def _pct(a, b) -> float:
    return round(100.0 * a / b, 4) if b else 0.0


def _column_profile(con, src: str, col: str, n: int) -> dict:
    nl = ", ".join(f"'{x}'" for x in NULL_LIKE)
    null, empty, ws, nulllike, uniq = con.execute(f"""
        SELECT count(*) FILTER (WHERE "{col}" IS NULL),
               count(*) FILTER (WHERE "{col}" = ''),
               count(*) FILTER (WHERE "{col}" <> '' AND trim("{col}") = ''),
               count(*) FILTER (WHERE lower(trim("{col}")) IN ({nl})),
               count(DISTINCT "{col}")
        FROM {src}""").fetchone()
    return {"null": null, "null_pct": _pct(null, n), "empty_string": empty, "whitespace_only": ws,
            "null_like_tokens": nulllike, "missing_total": null + empty + ws,
            "missing_total_pct": _pct(null + empty + ws, n), "unique_values": uniq}


def _text_stats(con, src: str, col: str) -> dict:
    r = con.execute(f"""
        WITH t AS (SELECT trim("{col}") v FROM {src} WHERE "{col}" IS NOT NULL AND trim("{col}") <> ''),
             w AS (SELECT length(v) l, len(string_split(regexp_replace(v, '\\s+', ' ', 'g'), ' ')) wc,
                          len(regexp_extract_all(v, '[0-9]+')) nt,
                          regexp_matches(v, '(^|[^0-9])[0-9]{{5,6}}([^0-9]|$)') postal,
                          regexp_matches(v, '[^\\x00-\\x7F]') non_ascii FROM t)
        SELECT count(*), min(l), max(l), avg(l), median(l), stddev_samp(l),
               min(wc), max(wc), avg(wc), median(wc),
               avg(CASE WHEN nt > 0 THEN 1 ELSE 0 END), avg(nt),
               avg(CASE WHEN postal THEN 1 ELSE 0 END), avg(CASE WHEN non_ascii THEN 1 ELSE 0 END)
        FROM w""").fetchone()
    keys = ["non_empty", "len_min", "len_max", "len_mean", "len_median", "len_std", "words_min", "words_max",
            "words_mean", "words_median", "share_with_numeric_token", "numeric_tokens_mean",
            "share_with_postal_like_code", "share_non_ascii"]
    return {k: (float(v) if v is not None else None) for k, v in zip(keys, r)}


def _profile_source(cfg: Config, con, t: str, man: dict) -> dict:
    src = f"read_parquet('{cfg.raw_dir / f'{t}.parquet'}')"
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    dup_rows = con.execute(f"""SELECT coalesce(sum(c - 1), 0) FROM (SELECT count(*) c FROM {src}
                               GROUP BY entity_id, business_name, business_address, country HAVING c > 1)""").fetchone()[0]
    dup_ids, dup_id_rows = con.execute(f"""SELECT count(*), coalesce(sum(c), 0) FROM (SELECT count(*) c FROM {src}
                                           GROUP BY entity_id HAVING c > 1)""").fetchone()
    raw_countries = con.execute(f"""SELECT coalesce(nullif(trim(country), ''), '<MISSING>') c, count(*) n
                                    FROM {src} GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 50""").fetchall()
    def top(col: str):
        return con.execute(f"""SELECT tok, count(*) n FROM (
        SELECT unnest(string_split(regexp_replace(lower("{col}"), '[^[:alnum:] ]', ' ', 'g'), ' ')) tok
        FROM (SELECT "{col}" FROM {src} USING SAMPLE 300000 ROWS (reservoir, 42))) WHERE tok <> ''
        GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 30""").fetchall()
    norm = pl.read_parquet(cfg.norm_dir / f"{t}.parquet",
                           columns=["normalized_name", "normalized_address", "normalized_country"])
    nn, na = pl.col("normalized_name"), pl.col("normalized_address")
    ne_n, ne_a = norm.filter(nn != ""), norm.filter(na != "")
    both = norm.filter((nn != "") & (na != ""))
    ncountry = norm["normalized_country"].fill_null("<missing>").value_counts(sort=True)
    m = man.get(t, {})
    return {
        "file": {"format": m.get("format"), "size_bytes": m.get("size_bytes"),
                 "size_mb": round((m.get("size_bytes") or 0) / 1e6, 2), "paths": [f["file"] for f in m.get("files", [])],
                 "columns": m.get("columns"), "dtypes": m.get("dtypes")},
        "rows": n, "columns": len(m.get("columns") or SOURCE_COLUMNS),
        "missing": {c: _column_profile(con, src, c, n) for c in SOURCE_COLUMNS},
        "duplicates": {
            "exact_duplicate_rows": int(dup_rows), "exact_duplicate_rows_pct": _pct(dup_rows, n),
            "duplicated_entity_ids": int(dup_ids), "rows_in_duplicate_id_groups": int(dup_id_rows),
            "normalized_name_collisions": ne_n.height - ne_n["normalized_name"].n_unique(),
            "normalized_address_collisions": ne_a.height - ne_a["normalized_address"].n_unique(),
            "normalized_name_and_address_collisions": both.height - both.select(nn, na).n_unique(),
        },
        "name": _text_stats(con, src, "business_name"),
        "address": _text_stats(con, src, "business_address"),
        "countries_raw": [(c, k, _pct(k, n)) for c, k in raw_countries],
        "countries_normalized": [(c, k, _pct(k, n)) for c, k in ncountry.iter_rows()],
        "n_countries_normalized": int(norm["normalized_country"].drop_nulls().n_unique()),
        "top_name_tokens": top("business_name"), "top_address_tokens": top("business_address"),
    }


def _ground_truth_profile(cfg: Config, con) -> dict:
    from data_loader import load, parse_gt
    raw = load(cfg, "train_ground_truth")
    gt = parse_gt(raw).with_columns(
        s2=pl.col("matched").list.eval(pl.element().str.starts_with("S2-").cast(pl.Int32)).list.sum(),
        s3=pl.col("matched").list.eval(pl.element().str.starts_with("S3-").cast(pl.Int32)).list.sum(),
        total=pl.col("matched").list.len())
    n = gt.height
    tot = gt["total"]
    multi = gt.filter(pl.col("total") > 1)
    hist = gt.group_by("total").len().sort("total")
    src = f"read_parquet('{cfg.raw_dir / 'train_ground_truth.parquet'}')"
    return {
        "rows": raw.height,
        "missing": {c: _column_profile(con, src, c, raw.height) for c in ("source1_entity_id", "matched_entity_ids")},
        "total_source1_entities": n,
        "zero_match": int((tot == 0).sum()), "zero_match_pct": _pct(int((tot == 0).sum()), n),
        "one_match": int((tot == 1).sum()), "one_match_pct": _pct(int((tot == 1).sum()), n),
        "multi_match": int((tot > 1).sum()), "multi_match_pct": _pct(int((tot > 1).sum()), n),
        "match_count_distribution": {int(k): int(v) for k, v in hist.iter_rows()},
        "entities_with_s2_matches": int((gt["s2"] > 0).sum()),
        "entities_with_s3_matches": int((gt["s3"] > 0).sum()),
        "entities_with_both_s2_and_s3": int(((gt["s2"] > 0) & (gt["s3"] > 0)).sum()),
        "entities_with_multiple_s2": int((gt["s2"] > 1).sum()),
        "entities_with_multiple_s3": int((gt["s3"] > 1).sum()),
        "total_s2_matches": int(gt["s2"].sum()), "total_s3_matches": int(gt["s3"].sum()),
        "mean_matches": float(tot.mean() or 0), "max_matches": int(tot.max() or 0),
        "multi_match_stats": {
            "mean": float(multi["total"].mean() or 0), "median": float(multi["total"].median() or 0),
            "max": int(multi["total"].max() or 0), "s2_share": _pct(int(multi["s2"].sum()), int(multi["total"].sum())),
            "s3_share": _pct(int(multi["s3"].sum()), int(multi["total"].sum())),
            "matched_to_both_sources": int(((multi["s2"] > 0) & (multi["s3"] > 0)).sum()),
        },
    }


def _cross_source_consistency(cfg: Config, sample: int = 500_000) -> dict:
    """How stable each field is between a Source-1 record and its true matches."""
    from data_loader import load, valid_gt_pairs
    pairs = valid_gt_pairs(cfg, load(cfg, "train_ground_truth"))
    if pairs.height > sample:
        pairs = pairs.sample(sample, seed=cfg.seed)
    cols = ["entity_id", "normalized_name", "name_core", "core_tokens", "address_word_tokens", "numeric_tokens",
            "postal_code_candidates", "building_number", "normalized_country", "normalized_address"]
    s1 = pl.read_parquet(cfg.norm_dir / "train_source1.parquet", columns=cols)
    cand = pl.concat([pl.read_parquet(cfg.norm_dir / f"train_source{i}.parquet", columns=cols) for i in (2, 3)])
    d = (pairs.join(s1.rename({c: f"a_{c}" for c in cols}), left_on="s1_id", right_on="a_entity_id")
              .join(cand.rename({c: f"b_{c}" for c in cols}), left_on="cand_id", right_on="b_entity_id"))
    if d.height == 0:
        return {}
    from rapidfuzz import fuzz, process
    jw = process.cpdist(d["a_normalized_name"].to_list(), d["b_normalized_name"].to_list(),
                        scorer=fuzz.token_set_ratio, workers=cfg.n_workers) / 100
    inter = lambda a, b: pl.col(a).list.set_intersection(pl.col(b)).list.len()  # noqa: E731
    both = lambda a, b: (pl.col(a).list.len() > 0) & (pl.col(b).list.len() > 0)  # noqa: E731
    cb = pl.col("a_normalized_country").is_not_null() & pl.col("b_normalized_country").is_not_null()
    agg = lambda df: df.select(  # noqa: E731
        pairs=pl.len(),
        country_both_present=cb.mean(),
        country_same_when_present=(pl.col("a_normalized_country") == pl.col("b_normalized_country")).filter(cb).mean(),
        name_exact=(pl.col("a_normalized_name") == pl.col("b_normalized_name")).mean(),
        core_name_exact=(pl.col("a_name_core") == pl.col("b_name_core")).mean(),
        shares_name_token=(inter("a_core_tokens", "b_core_tokens") > 0).mean(),
        shares_address_word_token=(inter("a_address_word_tokens", "b_address_word_tokens") > 0).filter(
            both("a_address_word_tokens", "b_address_word_tokens")).mean(),
        shares_numeric_token=(inter("a_numeric_tokens", "b_numeric_tokens") > 0).filter(
            both("a_numeric_tokens", "b_numeric_tokens")).mean(),
        postal_both_present=both("a_postal_code_candidates", "b_postal_code_candidates").mean(),
        postal_same_when_present=(inter("a_postal_code_candidates", "b_postal_code_candidates") > 0).filter(
            both("a_postal_code_candidates", "b_postal_code_candidates")).mean(),
        building_same_when_present=(pl.col("a_building_number") == pl.col("b_building_number")).filter(
            pl.col("a_building_number").is_not_null() & pl.col("b_building_number").is_not_null()).mean(),
        address_exact=(pl.col("a_normalized_address") == pl.col("b_normalized_address")).mean(),
        name_token_set_mean=pl.col("jw").mean(),
    ).row(0, named=True)
    d = d.with_columns(jw=pl.Series(jw), src=pl.col("cand_id").str.slice(0, 2))
    out = {"all": agg(d)}
    for s_ in ("S2", "S3"):
        sub = d.filter(pl.col("src") == s_)
        if sub.height:
            out[s_] = agg(sub)
    return {k: {kk: (round(float(vv), 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in out.items()}


def findings_to_decisions(rep: dict) -> list[dict]:
    """Turn measured dataset facts into explicit modelling decisions."""
    out = []
    add = lambda finding, decision: out.append({"finding": finding, "decision": decision})  # noqa: E731
    tabs = rep["tables"]
    for t in ("train_source2", "train_source3", "test_source2", "test_source3"):
        mp = tabs[t]["missing"]["business_address"]["missing_total_pct"]
        if mp > 5:
            add(f"{t}: {mp:.1f}% of addresses missing/empty",
                "Address similarities are NaN when missing (never 0); missing-address flags are features; "
                "name-based blocking keys do not need an address.")
    cmiss = max(tabs[t]["missing"]["country"]["missing_total_pct"] for t in tabs)
    if cmiss > 0:
        add(f"Up to {cmiss:.1f}% of records have no country",
            "Country-scoped blocking keys use '?'/'*' variants so missing-country records are never excluded; "
            "country_match is NaN when either side is missing.")
    raw_variants = max(len(tabs[t]["countries_raw"]) - tabs[t]["n_countries_normalized"] for t in tabs)
    if raw_variants > 0:
        add("Country values are spelled several ways (e.g. US / USA / United States)",
            "normalize_country maps aliases to one code; unknown values pass through (open set).")
    tr = {c for c, _, _ in tabs["train_source1"]["countries_normalized"]}
    te = {c for t in ("test_source1", "test_source2", "test_source3") for c, _, _ in tabs[t]["countries_normalized"]}
    unseen = sorted(te - tr - {"<missing>"})
    if unseen:
        add(f"Countries only in test: {unseen}",
            "No country-specific model or rule; features are country-agnostic, TF-IDF is fitted on each split's "
            "own unlabelled text, so unseen vocabulary still gets weights.")
    g = rep["ground_truth"]
    if g["zero_match_pct"] > 0:
        add(f"{g['zero_match_pct']:.1f}% of Source-1 entities have no match",
            "Explicit no-match decision: an entity gets an empty list when no candidate passes the rule; "
            "the rule is tuned on macro F0.5, where every correct empty list scores 1.")
    if g["multi_match_pct"] > 0:
        add(f"{g['multi_match_pct']:.1f}% of Source-1 entities have several matches (max {g['max_matches']})",
            "No argmax: every candidate passing the rule is returned; up to 25 candidates per source are kept.")
    if g["entities_with_multiple_s2"] or g["entities_with_multiple_s3"]:
        add("Some entities match several records of the same source",
            "Group-context features (same-name candidates, gap to best) let the model accept true duplicates.")
    dn = max(tabs[t]["duplicates"]["normalized_name_collisions"] / max(1, tabs[t]["rows"]) for t in tabs)
    if dn > 0.05:
        add(f"Up to {100 * dn:.1f}% of records share a normalised name with another record (chains, common names)",
            "Address evidence (postal, building number, address token-set) is weighted in candidate re-ranking "
            "and in the features; name-only keys are frequency-capped.")
    cs = rep.get("cross_source_consistency", {}).get("all", {})
    if cs:
        if (cs.get("country_same_when_present") or 0) > 0.95:
            add(f"Matches share the country {100 * cs['country_same_when_present']:.1f}% of the time when both have one",
                "Country is used to scope blocking keys (except exact-name keys) and as a feature, not as a sole criterion.")
        if (cs.get("name_exact") or 0) < 0.8:
            add(f"Only {100 * (cs.get('name_exact') or 0):.1f}% of true matches have identical normalised names",
                "Token, character n-gram and fuzzy retrieval blocking plus fuzzy/character-level similarity features.")
        if (cs.get("postal_same_when_present") or 0) > 0.8:
            add(f"Postal codes agree in {100 * cs['postal_same_when_present']:.1f}% of matches when both present",
                "Postal-based blocking keys (postal+name prefix, postal+building, token+postal) and postal features.")
    add("Dataset too large for S1 x (S2+S3)", "Multi-strategy blocking with hash joins; no Cartesian product.")
    add("Most in-block candidates are non-matches", "Train on blocked hard negatives, not random pairs.")
    return out


def run_eda(cfg: Config) -> dict:
    """Full data profile (DuckDB on Parquet + Polars) -> reports/eda.json and eda.md."""
    from data_loader import manifest
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={cfg.n_workers}")
    man = manifest(cfg)
    rep: dict = {"tables": {}}
    for t in SOURCE_TABLES:
        rep["tables"][t] = _profile_source(cfg, con, t, man)
        s = rep["tables"][t]
        log.info("EDA %-14s rows=%d dupRows=%d dupIDs=%d missing name/addr/country=%.2f%%/%.2f%%/%.2f%% "
                 "countries=%s", t, s["rows"], s["duplicates"]["exact_duplicate_rows"],
                 s["duplicates"]["duplicated_entity_ids"], s["missing"]["business_name"]["missing_total_pct"],
                 s["missing"]["business_address"]["missing_total_pct"], s["missing"]["country"]["missing_total_pct"],
                 s["countries_normalized"][:4])
    gtm = man.get("train_ground_truth", {})
    rep["ground_truth_file"] = {"format": gtm.get("format"), "size_mb": round((gtm.get("size_bytes") or 0) / 1e6, 2),
                                "columns": gtm.get("columns"), "dtypes": gtm.get("dtypes")}
    rep["ground_truth"] = _ground_truth_profile(cfg, con)
    rep["cross_source_consistency"] = _cross_source_consistency(cfg)
    # character classes (sample of names)
    from data_loader import load
    blocks = {}
    for t in ("train_source1", "test_source1"):
        names = load(cfg, t)["business_name"].drop_nulls()
        names = names.sample(min(50000, names.len()), seed=cfg.seed)
        c = Counter()
        for s_ in names:
            for ch in s_:
                o = ord(ch)
                c["ascii_letter" if ch.isascii() and ch.isalpha() else "digit" if ch.isdigit() else
                  "space" if ch.isspace() else "ascii_punct" if o < 128 else
                  "latin_ext" if o < 0x250 else "devanagari" if 0x900 <= o < 0x980 else "other_unicode"] += 1
        tot = sum(c.values()) or 1
        blocks[t] = {k: round(v / tot, 5) for k, v in c.most_common()}
    rep["character_distribution"] = blocks
    tr = {c for c, _, _ in rep["tables"]["train_source1"]["countries_normalized"]}
    te = {c for t in ("test_source1", "test_source2", "test_source3") for c, _, _ in rep["tables"][t]["countries_normalized"]}
    rep["countries_only_in_test"] = sorted(te - tr)
    rep["dataset_size"] = {t: rep["tables"][t]["rows"] for t in SOURCE_TABLES}
    rep["dataset_size"]["train_ground_truth"] = rep["ground_truth"]["rows"]
    rep["dataset_size"]["total_training_records"] = sum(rep["tables"][t]["rows"] for t in SOURCE_TABLES if t.startswith("train"))
    rep["dataset_size"]["total_test_records"] = sum(rep["tables"][t]["rows"] for t in SOURCE_TABLES if t.startswith("test"))
    rep["findings_to_decisions"] = findings_to_decisions(rep)
    (cfg.report_dir / "eda.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    _write_eda_md(cfg, rep)
    g = rep["ground_truth"]
    log.info("EDA ground truth: zero=%d (%.1f%%) one=%d multi=%d | S2 matches=%d S3=%d | both sources=%d",
             g["zero_match"], g["zero_match_pct"], g["one_match"], g["multi_match"], g["total_s2_matches"],
             g["total_s3_matches"], g["entities_with_both_s2_and_s3"])
    log.info("Countries only in test: %s", rep["countries_only_in_test"])
    return rep


def _write_eda_md(cfg: Config, rep: dict) -> None:
    T = rep["tables"]
    L = ["# EDA report", "", "## Files", "", "| Dataset | Format | Rows | Columns | File size (MB) | Exact duplicate rows | Duplicate IDs |",
         "|---|---|---|---|---|---|---|"]
    for t, s in T.items():
        L.append(f"| {t} | {s['file']['format']} | {s['rows']:,} | {s['columns']} | {s['file']['size_mb']} | "
                 f"{s['duplicates']['exact_duplicate_rows']:,} | {s['duplicates']['duplicated_entity_ids']:,} |")
    gf = rep["ground_truth_file"]
    L.append(f"| train_ground_truth | {gf['format']} | {rep['ground_truth']['rows']:,} | 2 | {gf['size_mb']} | – | – |")
    L += ["", "## Missing values (null / empty / whitespace-only → total %)", "",
          "| Dataset | entity_id | business_name | business_address | country |", "|---|---|---|---|---|"]
    for t, s in T.items():
        cell = lambda c: (f"{s['missing'][c]['null']:,} / {s['missing'][c]['empty_string']:,} / "  # noqa: E731
                          f"{s['missing'][c]['whitespace_only']:,} → {s['missing'][c]['missing_total_pct']:.2f}%")
        L.append(f"| {t} | {cell('entity_id')} | {cell('business_name')} | {cell('business_address')} | {cell('country')} |")
    L += ["", "## Duplicates after conservative normalisation", "",
          "| Dataset | name collisions | address collisions | name+address collisions |", "|---|---|---|---|"]
    for t, s in T.items():
        d = s["duplicates"]
        L.append(f"| {t} | {d['normalized_name_collisions']:,} | {d['normalized_address_collisions']:,} | "
                 f"{d['normalized_name_and_address_collisions']:,} |")
    L += ["", "## Business names / addresses", "",
          "| Dataset | field | len min / max / mean / median / std | words min / max / mean / median | numeric-token share | postal-like share |",
          "|---|---|---|---|---|---|"]
    for t, s in T.items():
        for f in ("name", "address"):
            x = s[f]
            fmt = lambda v: "–" if v is None else f"{v:.1f}"  # noqa: E731
            L.append(f"| {t} | {f} | {fmt(x['len_min'])} / {fmt(x['len_max'])} / {fmt(x['len_mean'])} / "
                     f"{fmt(x['len_median'])} / {fmt(x['len_std'])} | {fmt(x['words_min'])} / {fmt(x['words_max'])} / "
                     f"{fmt(x['words_mean'])} / {fmt(x['words_median'])} | {x['share_with_numeric_token'] or 0:.3f} | "
                     f"{x['share_with_postal_like_code'] or 0:.3f} |")
    L += ["", "## Countries (normalised)", "", "| Dataset | Country | Count | Percentage |", "|---|---|---|---|"]
    for t, s in T.items():
        for c, k, p in s["countries_normalized"][:12]:
            L.append(f"| {t} | {c} | {k:,} | {p:.2f}% |")
    L += ["", f"Countries only in test: {rep['countries_only_in_test']}", "", "## Ground truth", ""]
    g = rep["ground_truth"]
    L += [f"- Total Source 1 entities: {g['total_source1_entities']:,}",
          f"- Zero matches: {g['zero_match']:,} ({g['zero_match_pct']:.2f}%)",
          f"- Exactly one match: {g['one_match']:,} ({g['one_match_pct']:.2f}%)",
          f"- Multiple matches: {g['multi_match']:,} ({g['multi_match_pct']:.2f}%)",
          f"- Entities with Source 2 matches: {g['entities_with_s2_matches']:,}",
          f"- Entities with Source 3 matches: {g['entities_with_s3_matches']:,}",
          f"- Entities with both S2 and S3 matches: {g['entities_with_both_s2_and_s3']:,}",
          f"- Total S2 / S3 matches: {g['total_s2_matches']:,} / {g['total_s3_matches']:,}",
          f"- Multi-match mean / median / max: {g['multi_match_stats']['mean']:.2f} / "
          f"{g['multi_match_stats']['median']:.0f} / {g['multi_match_stats']['max']}",
          f"- Match-count distribution: {g['match_count_distribution']}", ""]
    cs = rep.get("cross_source_consistency", {})
    if cs:
        L += ["## Cross-source consistency of true matches", "", "| Measure | " + " | ".join(cs) + " |",
              "|---|" + "---|" * len(cs)]
        for k in cs["all"]:
            L.append(f"| {k} | " + " | ".join(str(cs[s_].get(k)) for s_ in cs) + " |")
        L.append("")
    L += ["## Findings → modelling decisions", ""]
    for f in rep["findings_to_decisions"]:
        L.append(f"- **{f['finding']}** → {f['decision']}")
    (cfg.report_dir / "eda.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_quality_report(cfg: Config) -> None:
    """Dataset quality report (sections: size, missing, duplicates, countries, ground truth,
    text, candidate generation, validation) filled from whatever reports exist so far."""
    R = cfg.report_dir
    rd = lambda n: json.loads((R / n).read_text(encoding="utf-8")) if (R / n).exists() else None  # noqa: E731
    eda, rec, fin = rd("eda.json"), rd("candidate_recall_train.json"), rd("final_validation.json")
    P = "pending (stage not run yet)"
    L = ["# Dataset quality report", "", "## Dataset size", ""]
    if eda:
        for k, v in eda["dataset_size"].items():
            L.append(f"- {k}: {v:,}")
        L += ["", "## Missing values (null + empty + whitespace-only)", ""]
        for t, s in eda["tables"].items():
            L.append(f"- **{t}**: " + ", ".join(f"{c}={s['missing'][c]['missing_total']:,} "
                                                f"({s['missing'][c]['missing_total_pct']:.2f}%)" for c in SOURCE_COLUMNS))
        gm = eda["ground_truth"]["missing"]
        L.append("- **train_ground_truth**: " + ", ".join(f"{c}={v['missing_total']:,}" for c, v in gm.items())
                 + " (empty matched_entity_ids = zero-match entity)")
        L += ["", "## Duplicate analysis", ""]
        for t, s in eda["tables"].items():
            d = s["duplicates"]
            L.append(f"- **{t}**: exact duplicate records={d['exact_duplicate_rows']:,}, duplicate entity IDs="
                     f"{d['duplicated_entity_ids']:,}, duplicate normalised names={d['normalized_name_collisions']:,}, "
                     f"duplicate normalised addresses={d['normalized_address_collisions']:,}")
        L += ["", "## Country distribution (normalised codes)", "",
              "| Dataset | India (in) | USA (us) | France (fr) | Other | Missing |", "|---|---|---|---|---|---|"]
        for t, s in eda["tables"].items():
            cn = {c: k for c, k, _ in s["countries_normalized"]}
            other = sum(k for c, k in cn.items() if c not in ("in", "us", "fr", "<missing>"))
            L.append(f"| {t} | {cn.get('in', 0):,} | {cn.get('us', 0):,} | {cn.get('fr', 0):,} | {other:,} | "
                     f"{cn.get('<missing>', 0):,} |")
        g = eda["ground_truth"]
        L += ["", "## Ground truth", "", f"- Zero-match entities: {g['zero_match']:,} ({g['zero_match_pct']:.2f}%)",
              f"- Single-match entities: {g['one_match']:,} ({g['one_match_pct']:.2f}%)",
              f"- Multi-match entities: {g['multi_match']:,} ({g['multi_match_pct']:.2f}%)",
              f"- Source 2 matches: {g['total_s2_matches']:,}", f"- Source 3 matches: {g['total_s3_matches']:,}",
              f"- Entities having both S2 and S3 matches: {g['entities_with_both_s2_and_s3']:,}",
              "", "## Text statistics (all sources)", ""]
        for t, s in eda["tables"].items():
            L.append(f"- **{t}**: name length mean {s['name']['len_mean'] or 0:.1f} / median {s['name']['len_median'] or 0:.0f}; "
                     f"address length mean {s['address']['len_mean'] or 0:.1f} / median {s['address']['len_median'] or 0:.0f}")
    else:
        L.append(P)
    L += ["", "## Candidate generation (train Source 1, blocking + capping)", ""]
    if rec:
        L += [f"- Total possible pairs: {rec['cartesian_pairs']:,}", f"- Generated candidate pairs: {rec['total_pairs']:,}",
              f"- Candidate recall: {rec['candidate_recall']:.4f} (S2 {rec['by_source']['S2']['recall'] or 0:.4f}, "
              f"S3 {rec['by_source']['S3']['recall'] or 0:.4f}); before capping {rec['pre_cap_recall'] or 0:.4f}",
              f"- Candidate reduction ratio: {rec['reduction_ratio']:.6f}",
              f"- Average / median / maximum candidates per Source 1: {rec['avg_candidates_per_s1']:.1f} / "
              f"{rec['median_candidates_per_s1']:.0f} / {rec['max_candidates_per_s1']}"]
    else:
        L.append(P)
    L += ["", "## Validation (held-out 20% of Source 1 entities)", ""]
    if fin:
        e = fin["ensemble"]
        L += [f"- Objective: {fin['objective']}", f"- Macro F0.5 (challenge metric): {e['macro_f0_5']:.4f}",
              f"- Precision (micro): {e['micro_precision']:.4f}", f"- Recall (micro): {e['micro_recall']:.4f}",
              f"- F0.5 (micro): {e['micro_f0_5']:.4f}", f"- F1 (micro): {e['micro_f1']:.4f}",
              f"- Optimal threshold: {fin['rule']['threshold']} (relative ratio {fin['rule']['relative_ratio']}, "
              f"max matches {fin['rule']['max_matches'] or 'unlimited'})",
              f"- False merges: {e['false_merges']:,}; missed matches: {e['missed_matches']:,}",
              f"- Zero-match entities correct: {e['zero_match_correct']:,}/{e['zero_match_total']:,}",
              f"- One-match entities exactly right: {e['one_match_exact']:,}/{e['one_match_total']:,}",
              f"- Multi-match entities exactly right: {e['multi_match_exact']:,}/{e['multi_match_total']:,}",
              f"- Validation candidate recall: {fin['validation_candidate_recall']:.4f}",
              f"- For reference at threshold 0.50: macro F0.5 {fin['ensemble_at_threshold_0_5']['macro_f0_5']:.4f}, "
              f"false merges {fin['ensemble_at_threshold_0_5']['false_merges']:,}"]
    else:
        L.append(P)
    (R / "dataset_quality_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(cfg: Config, val: pl.DataFrame, pred_mask: np.ndarray, val_gt: pl.DataFrame,
                   s1: pl.DataFrame, cand: pl.DataFrame, n: int = 300) -> dict:
    """Write FP / FN samples with the raw records and summarise error causes."""
    v = val.with_columns(pred=pl.Series(pred_mask))
    fp = v.filter(pl.col("pred") & (pl.col("label") == 0))
    tp = v.filter(pl.col("pred") & (pl.col("label") == 1)).select("s1_id", "cand_id")
    fn_all = val_gt.join(tp, on=["s1_id", "cand_id"], how="anti")
    in_cands = v.filter(pl.col("label") == 1).select("s1_id", "cand_id", "p_final")
    fn = fn_all.join(in_cands, on=["s1_id", "cand_id"], how="left").with_columns(
        cause=pl.when(pl.col("p_final").is_null()).then(pl.lit("missed_by_blocking")).otherwise(pl.lit("below_threshold")))
    s1r = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s1_name"),
                    pl.col("business_address").alias("s1_address"), pl.col("country").alias("s1_country"))
    cr = cand.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("cand_name"),
                     pl.col("business_address").alias("cand_address"), pl.col("country").alias("cand_country"))
    fp_s = fp.sort("p_final", descending=True).head(n).select("s1_id", "cand_id", "p_final", "name_jw", "addr_tset") \
             .join(s1r, on="s1_id", how="left").join(cr, on="cand_id", how="left")
    fn_s = fn.head(n).join(s1r, on="s1_id", how="left").join(cr, on="cand_id", how="left")
    fp_s.write_csv(cfg.report_dir / "error_analysis_false_positives.tsv", separator="\t")
    fn_s.write_csv(cfg.report_dir / "error_analysis_false_negatives.tsv", separator="\t")
    bucket = lambda c: (pl.when(pl.col(c) >= .9).then(pl.lit(">=0.9")).when(pl.col(c) >= .7).then(pl.lit("0.7-0.9"))  # noqa: E731
                        .otherwise(pl.lit("<0.7")))
    summary = {
        "false_positives": fp.height,
        "false_negatives": fn.height,
        "fn_by_cause": dict(fn.group_by("cause").len().iter_rows()),
        "fp_by_name_similarity": dict(fp.with_columns(b=bucket("name_jw")).group_by("b").len().iter_rows()),
        "fp_same_postal": int(fp["postal_match"].fill_null(0).sum()) if fp.height else 0,
        "fp_country_mismatch": int((fp["country_match"] == 0).sum()) if fp.height else 0,
        "fp_by_source": dict(fp.group_by("source").len().iter_rows()),
        "fn_by_source": dict(fn.with_columns(src=pl.col("cand_id").str.slice(0, 2)).group_by("src").len().iter_rows()),
    }
    (cfg.report_dir / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Error analysis: %s", summary)
    return summary

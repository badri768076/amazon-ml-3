"""Pipeline entry point.

    python src/main.py --stage eda | preprocess | blocking | train | evaluate |
                               inference | submission | compliance | all

Data comes only from the supplied Kaggle dataset (config.DATA_ROOT). Every stage
checkpoints to ``artifacts/`` and skips work that already exists (``--force``
recomputes), so a failed run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from config import TEST_TABLES, TRAIN_TABLES, Config, set_seeds, setup_logging  # noqa: E402

log = logging.getLogger("ber.main")
STAGES = ["eda", "preprocess", "blocking", "train", "evaluate", "inference", "submission"]


# --------------------------------------------------------------------------- #
def stage_eda(cfg: Config) -> None:
    from data_loader import materialise_raw, validate_tables
    from evaluation import run_eda, write_quality_report
    from preprocessing import preprocess_table
    materialise_raw(cfg)
    validate_tables(cfg)
    for t in TRAIN_TABLES + TEST_TABLES:     # normalised views are needed for collision / consistency EDA
        if not t.endswith("ground_truth"):
            preprocess_table(cfg, t)
    run_eda(cfg)
    write_quality_report(cfg)


def stage_preprocess(cfg: Config) -> None:
    from preprocessing import preprocess_table
    for t in TRAIN_TABLES + TEST_TABLES:
        if not t.endswith("ground_truth"):
            preprocess_table(cfg, t)


def stage_blocking(cfg: Config) -> None:
    from candidate_generation import candidate_recall, generate_candidates
    from training_pairs import select_train_entities
    ids, gtp = select_train_entities(cfg)
    generate_candidates(cfg, "train", ids, gt=gtp)
    candidate_recall(cfg, "train", gtp)
    generate_candidates(cfg, "test")
    from evaluation import write_quality_report
    write_quality_report(cfg)


def _evaluator(cfg: Config, df: pl.DataFrame, role: str):
    from evaluation import GroupEvaluator
    gt = pl.read_parquet(cfg.art / "training" / f"{role}_gt.parquet")
    ents = pl.read_parquet(cfg.art / "training" / f"{role}_entities.parquet")["s1_id"].to_list()
    counts = dict.fromkeys(ents, 0)
    counts.update(dict(gt.group_by("s1_id").len().iter_rows()))
    return GroupEvaluator(df["s1_idx"].to_numpy(), df["label"].to_numpy(), df["s1_id"].to_numpy(), counts,
                          cfg.objective), gt


def _fmt(m: dict) -> str:
    return (f"macro F0.5={m['macro_f0_5']:.4f} | micro P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} "
            f"F0.5={m['micro_f0_5']:.4f} F1={m['micro_f1']:.4f} | false merges={m['false_merges']} "
            f"missed={m['missed_matches']} | zero-match {m['zero_match_correct']}/{m['zero_match_total']} "
            f"one-match {m['one_match_exact']}/{m['one_match_total']} "
            f"multi-match {m['multi_match_exact']}/{m['multi_match_total']}")


def stage_train(cfg: Config) -> None:
    from candidate_generation import entity_tables
    from feature_engineering import build_feature_matrix
    from models import PairModels, save_importance, to_xy
    from threshold_optimization import best_threshold
    from training_pairs import build_training_sets, select_train_entities

    done = [cfg.art / "models" / "selection.joblib", cfg.art / "predictions" / "val_probs.parquet",
            cfg.art / "predictions" / "inner_probs.parquet", cfg.report_dir / "model_validation.json"]
    if all(p.exists() for p in done) and not cfg.force:
        log.info("[checkpoint] trained models and validation predictions exist")
        return
    s1, cand = entity_tables(cfg, "train")
    pairs = pl.read_parquet(cfg.art / "candidates" / "train_pairs.parquet")
    feat_dir = build_feature_matrix(cfg, "train", pairs, s1, cand)
    _, gtp = select_train_entities(cfg)
    sets = build_training_sets(cfg, feat_dir, s1, cand, gtp)
    Xtr, ytr = to_xy(sets["train"])
    Xin, yin = to_xy(sets["inner"])
    models = PairModels(cfg).fit(Xtr, ytr, Xin, yin)
    models.save("selection")
    save_importance(cfg, models)

    report = {}
    for role in ("inner", "val"):
        df = sets[role].sort("s1_idx", "cand_idx")
        X, _ = to_xy(df)
        probs = models.predict(X)
        pl.DataFrame({"s1_idx": df["s1_idx"], "cand_idx": df["cand_idx"], "s1_id": df["s1_id"],
                      "cand_id": df["cand_id"], "label": df["label"],
                      **{f"p_{k}": v for k, v in probs.items()}}).write_parquet(
            cfg.path("predictions", f"{role}_probs.parquet"))
        if role == "val":
            ev, _ = _evaluator(cfg, df, "val")
            for name, p in probs.items():
                t, m = best_threshold(p, ev, cfg)
                report[name] = {"best_threshold": t, **m}
                log.info("Validation %-5s t=%.2f %s%s", name, t, _fmt(m), "  (baseline)" if name == "lr" else "")
    (cfg.report_dir / "model_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def stage_evaluate(cfg: Config) -> None:
    from candidate_generation import entity_tables
    from ensemble import combine, select_ensemble
    from evaluation import error_analysis
    from models import ENSEMBLE_MODELS
    from threshold_optimization import decision_mask, save_rule, threshold_table

    val = pl.read_parquet(cfg.art / "predictions" / "val_probs.parquet")
    inner = pl.read_parquet(cfg.art / "predictions" / "inner_probs.parquet")
    ev, val_gt = _evaluator(cfg, val, "val")
    vp = {m: val[f"p_{m}"].to_numpy() for m in ENSEMBLE_MODELS}
    ip = {m: inner[f"p_{m}"].to_numpy() for m in ENSEMBLE_MODELS}
    res = select_ensemble(vp, ev, ip, inner["label"].to_numpy(), cfg)
    rule = dict(res["rule"])
    rule.update({"weights": res["weights"], "calibration": res["calibration"], "objective": cfg.objective,
                 "selected_on": "validation"})
    p_final = combine(res["calibrator"].transform(vp), res["weights"])
    table = threshold_table(p_final, ev, cfg, rule["relative_ratio"], rule["max_matches"])
    save_rule(cfg, rule, table)
    pl.DataFrame(res["rule_rows"]).write_csv(cfg.report_dir / "decision_rule_search.tsv", separator="\t")

    mask = decision_mask(p_final, rule["threshold"], rule["relative_ratio"], rule["max_matches"],
                         ev.group_max(p_final), ev.group_rank(p_final))
    final = ev.evaluate(mask)
    base = json.loads((cfg.report_dir / "model_validation.json").read_text(encoding="utf-8"))
    at_05 = ev.evaluate(p_final >= 0.5)
    summary = {"ensemble": final, "ensemble_at_threshold_0_5": at_05,
               "rule": {k: rule[k] for k in ("threshold", "relative_ratio", "max_matches")},
               "objective": cfg.objective, "weights": res["weights"], "calibration": res["calibration"],
               "validation_candidate_recall": float(val["label"].sum()) / max(1, val_gt.height),
               "single_models": base}
    (cfg.report_dir / "final_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("FINAL VALIDATION (ensemble): %s", _fmt(final))
    log.info("  at t=0.50 for reference: %s", _fmt(at_05))
    log.info("  LR baseline: macro F0.5=%.4f", base["lr"]["macro_f0_5"])
    s1, cand = entity_tables(cfg, "train")
    error_analysis(cfg, val.with_columns(p_final=pl.Series(p_final)).join(
        pl.read_parquet(cfg.art / "training" / "val_pairs.parquet").select(
            "s1_idx", "cand_idx", "name_jw", "addr_tset", "postal_match", "country_match", "source"),
        on=["s1_idx", "cand_idx"], how="left"), mask, val_gt, s1, cand)
    from evaluation import write_quality_report
    write_quality_report(cfg)


def stage_inference(cfg: Config) -> None:
    from inference import predict_test
    predict_test(cfg)


def stage_submission(cfg: Config) -> None:
    from evaluation import write_quality_report
    from submission import compliance_check, run_official_validator, validate_submission, write_submission
    scored = pl.read_parquet(cfg.art / "predictions" / "test_scored.parquet")
    mp, cp = write_submission(cfg, scored)
    rep = validate_submission(cfg, mp, cp)
    official = run_official_validator(cfg, mp, cp)
    write_quality_report(cfg)
    compliance_check(cfg, rep, official)


def stage_compliance(cfg: Config) -> None:
    """Re-validate existing output files and re-run the compliance checklist."""
    from pathlib import Path
    from submission import compliance_check, run_official_validator, validate_submission
    mp, cp = Path(cfg.output_dir) / "matching_results.tsv", Path(cfg.output_dir) / "candidate_pairs.tsv"
    compliance_check(cfg, validate_submission(cfg, mp, cp), run_official_validator(cfg, mp, cp))


RUNNERS = {"eda": stage_eda, "preprocess": stage_preprocess, "blocking": stage_blocking, "train": stage_train,
           "evaluate": stage_evaluate, "inference": stage_inference, "submission": stage_submission,
           "compliance": stage_compliance}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    ap.add_argument("--stage", required=True, choices=STAGES + ["compliance", "all"])
    ap.add_argument("--data-root", "--data-dir", dest="data_root",
                    help="dataset directory (default: the supplied Kaggle dataset under /kaggle/input)")
    ap.add_argument("--artifacts-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--max-train-s1", type=int, help="cap on labelled train Source-1 entities")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true", help="recompute checkpoints of the selected stage(s)")
    a = ap.parse_args(argv)

    cfg = Config()
    for k in ("data_root", "output_dir", "max_train_s1", "seed"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.artifacts_dir:
        cfg.artifacts_dir = a.artifacts_dir
    if a.workers:
        cfg.n_workers = a.workers
    cfg.force = a.force
    setup_logging(cfg)
    set_seeds(cfg.seed)
    cfg.path("config_used.json").write_text(cfg.to_json(), encoding="utf-8")
    log.info("Dataset: %s | artifacts: %s | output: %s", cfg.data_root, cfg.artifacts_dir, cfg.output_dir)

    for st in (STAGES if a.stage == "all" else [a.stage]):
        t = time.time()
        log.info("===== STAGE %s =====", st.upper())
        RUNNERS[st](cfg)
        log.info("===== STAGE %s done in %.1fs =====", st.upper(), time.time() - t)


if __name__ == "__main__":
    main()

"""Dataset discovery, loading and validation.

Single data source: the supplied Kaggle dataset at ``config.DATA_ROOT``
(``/kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml``).
Nothing is downloaded and no external business information is used.

* Files are discovered recursively (``DATA_ROOT.rglob("*")``) and mapped to the
  seven logical tables whatever the sub-folder layout
  (``train_source1.tsv``, ``train/source1.parquet``, ``test/test_source3.csv``, ...).
* The reader is chosen from the file extension (.tsv / .csv / .parquet).
* Only the required columns are kept, every value is read as text (IDs, PIN and
  ZIP codes keep their leading zeros), and empty strings are preserved so EDA can
  tell *null* from *empty* from *whitespace-only*.
* Each table is converted once to ``artifacts/raw/<table>.parquet``; later stages
  read those lazily with Polars / DuckDB, one table at a time.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from config import ALL_TABLES, GT_COLUMNS, SOURCE_COLUMNS, SOURCE_PREFIX, TEST_TABLES, TRAIN_TABLES, Config

log = logging.getLogger("ber.data")

SUPPORTED = {".tsv", ".csv", ".parquet", ".txt"}
_SHARD_RE = re.compile(r"[-_](\d{5})[-_]of[-_](\d{5})$")
_ID_SPLIT_RE = r"[,;|\s]+"


class DataError(RuntimeError):
    """Raised for missing files, missing columns, invalid schemas or corrupt data."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_files(cfg: Config) -> list[Path]:
    root = Path(cfg.data_root)
    if not root.exists():
        raise DataError(f"Dataset directory not found: {root}. Attach the supplied Kaggle dataset "
                        "to the notebook, or point BER_DATA_ROOT / --data-root at a local copy.")
    return sorted(p for p in root.rglob("*") if p.is_file())


def _required_columns(table: str) -> list[str]:
    return GT_COLUMNS if table.endswith("ground_truth") else SOURCE_COLUMNS


def canonical_table_name(path: str | Path) -> str | None:
    """Map a file path to one of the seven logical table names (or None)."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED:
        return None
    stem = _SHARD_RE.sub("", p.stem).lower().replace("-", "_").replace(" ", "_")
    aliases = {"groundtruth": "ground_truth", "gt": "ground_truth", "labels": "ground_truth",
               "ground_truths": "ground_truth"}
    for split in ("train", "test"):
        if stem.startswith(f"{split}_"):
            rest = stem[len(split) + 1:]
            stem = f"{split}_{aliases.get(rest, rest)}"
    if stem in ALL_TABLES:
        return stem
    base = aliases.get(stem, stem)
    parents = [x.lower() for x in p.parts[:-1]]
    for split in ("train", "test"):
        if any(x == split or x.startswith(f"{split}_") or x.endswith(f"_{split}") for x in parents[-2:]):
            name = f"{split}_{base}"
            if name in ALL_TABLES:
                return name
    return None


def discover_tables(cfg: Config) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for f in discover_files(cfg):
        name = canonical_table_name(f.relative_to(cfg.data_root))
        if name:
            found.setdefault(name, []).append(f)
        else:
            log.debug("Ignoring non-table file %s", f)
    for t, files in found.items():
        # prefer parquet when the same table exists in several formats
        exts = {f.suffix.lower() for f in files}
        if len(exts) > 1:
            keep = ".parquet" if ".parquet" in exts else sorted(exts)[0]
            found[t] = [f for f in files if f.suffix.lower() == keep]
            log.warning("%s present in several formats %s - using %s", t, sorted(exts), keep)
    return found


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _match_columns(available: list[str], required: list[str], table: str) -> dict[str, str]:
    """Case/whitespace-insensitive column matching; raises on missing columns."""
    lookup = {c.strip().lower(): c for c in available}
    mapping, missing = {}, []
    for col in required:
        if col in lookup:
            mapping[lookup[col]] = col
        else:
            missing.append(col)
    if missing:
        raise DataError(f"{table}: missing required column(s) {missing}; found {available}")
    return mapping


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            n += block.count(b"\n")
    return n


def read_table_file(path: Path, table: str) -> tuple[pl.DataFrame, dict]:
    """Read one file with only the required columns, all values as text."""
    req = _required_columns(table)
    ext = path.suffix.lower()
    info = {"file": str(path), "format": ext.lstrip("."), "size_bytes": path.stat().st_size}
    try:
        if ext == ".parquet":
            pf = pq.ParquetFile(path)
            info["original_columns"] = pf.schema_arrow.names
            info["original_dtypes"] = {f.name: str(f.type) for f in pf.schema_arrow}
            mapping = _match_columns(pf.schema_arrow.names, req, table)
            df = pl.from_arrow(pf.read(columns=list(mapping)))
        else:
            sep = "," if ext == ".csv" else "\t"
            opts = dict(separator=sep, infer_schema_length=0, null_values=None, encoding="utf8-lossy",
                        truncate_ragged_lines=False)
            try:
                df = pl.read_csv(path, quote_char='"', **opts)
            except Exception:  # stray quotes inside unquoted fields
                log.warning("%s: retrying without quote handling", path.name)
                df = pl.read_csv(path, quote_char=None, **opts)
            info["original_columns"] = df.columns
            sample = pl.read_csv(path, separator=sep, n_rows=2000, quote_char='"', encoding="utf8-lossy",
                                 infer_schema_length=2000, truncate_ragged_lines=True, ignore_errors=True)
            info["original_dtypes"] = {c: str(t) for c, t in sample.schema.items()}
            mapping = _match_columns(df.columns, req, table)
            # a text file has no null marker: an absent value is an empty string
            df = df.select(list(mapping)).with_columns(pl.all().fill_null(""))
            lines = _count_lines(path)
            info["text_lines"] = lines
            if abs(lines - 1 - df.height) > 1:
                log.warning("%s: %d rows parsed from %d lines (multi-line or quoted fields?)",
                            path.name, df.height, lines)
    except DataError:
        raise
    except Exception as exc:  # corrupted file, unreadable encoding, ragged rows, ...
        raise DataError(f"{table}: failed to read {path}: {exc}") from exc
    df = df.rename(mapping).with_columns([pl.col(c).cast(pl.Utf8) for c in req])
    info.update({"rows": df.height, "columns_used": req})
    return df, info


def _standardise(df: pl.DataFrame, table: str) -> pl.DataFrame:
    if table.endswith("ground_truth"):
        return df.with_columns(pl.col("source1_entity_id").str.strip_chars(),
                               pl.col("matched_entity_ids").fill_null("").str.strip_chars())
    return df.with_columns(pl.col("entity_id").str.strip_chars())


def materialise_raw(cfg: Config, tables: list[str] | None = None) -> dict[str, Path]:
    """Convert the supplied files to column-pruned Parquet once (checkpointed)."""
    tables = tables or ALL_TABLES
    out = {t: cfg.raw_dir / f"{t}.parquet" for t in tables}
    todo = [t for t in tables if cfg.force or not out[t].exists()]
    if not todo:
        return out
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    found = discover_tables(cfg)
    log.info("Dataset root %s -> tables found: %s", cfg.data_root, sorted(found))
    manifest_path = cfg.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for t in todo:
        if t not in found:
            raise DataError(f"Required table '{t}' not found under {cfg.data_root}")
        frames, infos = [], []
        for f in sorted(found[t]):
            df, info = read_table_file(f, t)
            frames.append(df)
            infos.append(info)
        df = _standardise(pl.concat(frames) if len(frames) > 1 else frames[0], t)
        df.write_parquet(out[t], compression="zstd")
        manifest[t] = {"files": infos, "rows": df.height, "size_bytes": sum(i["size_bytes"] for i in infos),
                       "format": infos[0]["format"], "columns": infos[0].get("original_columns"),
                       "dtypes": infos[0].get("original_dtypes")}
        log.info("Loaded %-20s rows=%10d  (%s, %.1f MB)", t, df.height, infos[0]["format"],
                 manifest[t]["size_bytes"] / 1e6)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def manifest(cfg: Config) -> dict:
    p = cfg.raw_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def scan(cfg: Config, table: str) -> pl.LazyFrame:
    """Lazy Polars scan of a materialised table."""
    p = cfg.raw_dir / f"{table}.parquet"
    if not p.exists():
        materialise_raw(cfg, [table])
    return pl.scan_parquet(p)


def load(cfg: Config, table: str) -> pl.DataFrame:
    return scan(cfg, table).collect()


def parse_gt(gt: pl.DataFrame) -> pl.DataFrame:
    """Ground truth -> (source1_entity_id, matched: list[str]) with empties removed."""
    return gt.with_columns(
        pl.col("matched_entity_ids").fill_null("").str.replace_all(_ID_SPLIT_RE, ",").str.split(",")
    ).with_columns(
        pl.col("matched_entity_ids").list.eval(
            pl.element().str.strip_chars().filter(pl.element().str.len_chars() > 0)).alias("matched")
    ).select("source1_entity_id", "matched")


def gt_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Exploded ground-truth pairs (s1_id, cand_id), de-duplicated."""
    return parse_gt(gt).explode("matched").drop_nulls("matched") \
        .rename({"source1_entity_id": "s1_id", "matched": "cand_id"}).unique()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_tables(cfg: Config) -> dict:
    """Schema / ID / ground-truth checks on all seven tables -> validation_report.json.

    Errors (stop the run): empty table, missing IDs, IDs with the wrong or an
    unexpected prefix, duplicate ground-truth rows. Duplicate entity IDs are an
    error only with ``strict_validation`` (they are otherwise reported and handled).
    """
    materialise_raw(cfg)
    report: dict = {"tables": {}, "ground_truth": {}, "errors": [], "warnings": []}
    for t in ALL_TABLES:
        if t.endswith("ground_truth"):
            continue
        df = load(cfg, t)
        src = t.split("_")[1]
        prefix = SOURCE_PREFIX[src]
        ids = df["entity_id"].fill_null("")
        pref = ids.str.extract(r"^(S\d+)-", 1)
        dup_groups = df.group_by("entity_id").len().filter(pl.col("len") > 1)
        stats = {
            "rows": df.height,
            "unique_entity_ids": ids.n_unique(),
            "missing_entity_ids": int((ids.str.strip_chars() == "").sum()),
            "duplicate_entity_ids": dup_groups.height,
            "rows_in_duplicate_id_groups": int(dup_groups["len"].sum() or 0),
            "invalid_prefix": int((~ids.str.starts_with(prefix)).sum()),
            "prefix_counts": dict(pref.fill_null("<none>").value_counts().iter_rows()),
            "ids_from_other_sources": int(pref.is_in([p.rstrip("-") for s, p in SOURCE_PREFIX.items()
                                                      if s != src]).sum()),
        }
        if df.height == 0:
            report["errors"].append(f"{t}: table is empty")
        if stats["missing_entity_ids"]:
            report["errors"].append(f"{t}: {stats['missing_entity_ids']} missing entity_id values")
        if stats["invalid_prefix"]:
            report["errors"].append(f"{t}: {stats['invalid_prefix']} ids without prefix {prefix} "
                                    f"(prefixes seen: {stats['prefix_counts']})")
        if stats["duplicate_entity_ids"]:
            msg = (f"{t}: {stats['duplicate_entity_ids']} duplicated entity_id values "
                   f"({stats['rows_in_duplicate_id_groups']} rows)")
            (report["errors"] if cfg.strict_validation else report["warnings"]).append(msg)
        report["tables"][t] = stats
        log.info("Validated %-14s rows=%d unique_ids=%d bad_prefix=%d dup_ids=%d", t, df.height,
                 stats["unique_entity_ids"], stats["invalid_prefix"], stats["duplicate_entity_ids"])

    s1 = load(cfg, "train_source1").select(pl.col("entity_id").alias("id")).unique()
    s2 = load(cfg, "train_source2").select(pl.col("entity_id").alias("id")).unique()
    s3 = load(cfg, "train_source3").select(pl.col("entity_id").alias("id")).unique()
    gt = load(cfg, "train_ground_truth")
    parsed = parse_gt(gt)
    pairs = parsed.explode("matched").drop_nulls("matched").rename({"matched": "id"})
    is2, is3 = pl.col("id").str.starts_with("S2-"), pl.col("id").str.starts_with("S3-")
    lens = parsed["matched"].list.len()
    g = {
        "rows": gt.height,
        "unique_s1": parsed["source1_entity_id"].n_unique(),
        "duplicate_s1_rows": gt.height - parsed["source1_entity_id"].n_unique(),
        "s1_not_in_train_source1": parsed.select(pl.col("source1_entity_id").alias("id"))
                                         .join(s1, on="id", how="anti").height,
        "train_source1_without_gt_row": s1.join(parsed.select(pl.col("source1_entity_id").alias("id")),
                                                on="id", how="anti").height,
        "s2_ids_not_in_train_source2": pairs.filter(is2).join(s2, on="id", how="anti").height,
        "s3_ids_not_in_train_source3": pairs.filter(is3).join(s3, on="id", how="anti").height,
        "source1_ids_as_targets": pairs.filter(pl.col("id").str.starts_with("S1-")).height,
        "unexpected_target_prefix": pairs.filter(~is2 & ~is3).height,
        "rows_with_duplicate_match_ids": parsed.filter(
            pl.col("matched").list.len() != pl.col("matched").list.unique().list.len()).height,
        "self_matches": pairs.filter(pl.col("source1_entity_id") == pl.col("id")).height,
        "target_ids_used_by_multiple_s1": pairs.group_by("id").len().filter(pl.col("len") > 1).height,
        "zero_match": int((lens == 0).sum()), "one_match": int((lens == 1).sum()),
        "multi_match": int((lens > 1).sum()),
    }
    if g["duplicate_s1_rows"]:
        report["errors"].append(f"train_ground_truth: {g['duplicate_s1_rows']} duplicate Source-1 rows")
    for k in ("s1_not_in_train_source1", "s2_ids_not_in_train_source2", "s3_ids_not_in_train_source3",
              "source1_ids_as_targets", "unexpected_target_prefix", "self_matches",
              "rows_with_duplicate_match_ids"):
        if g[k]:
            report["warnings"].append(f"train_ground_truth: {k} = {g[k]} (such targets are ignored)")
    report["ground_truth"] = g
    log.info("Ground truth: rows=%d zero=%d one=%d multi=%d | invalid S2=%d S3=%d S1-as-target=%d",
             g["rows"], g["zero_match"], g["one_match"], g["multi_match"], g["s2_ids_not_in_train_source2"],
             g["s3_ids_not_in_train_source3"], g["source1_ids_as_targets"])

    (cfg.report_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for w in report["warnings"]:
        log.warning(w)
    if report["errors"]:
        for e in report["errors"]:
            log.error(e)
        raise DataError("Data validation failed - see artifacts/reports/validation_report.json")
    log.info("Data validation passed (%d warnings)", len(report["warnings"]))
    return report


def valid_gt_pairs(cfg: Config, gt: pl.DataFrame) -> pl.DataFrame:
    """GT pairs restricted to targets that exist in train Source 2 / Source 3."""
    cands = pl.concat([load(cfg, "train_source2").select("entity_id"),
                       load(cfg, "train_source3").select("entity_id")]).rename({"entity_id": "cand_id"})
    return gt_pairs(gt).join(cands.unique(), on="cand_id", how="semi")


def resolve_gt_policy(cfg: Config, gt: pl.DataFrame) -> str:
    """How Source-1 train entities absent from the ground truth are treated.

    auto: if the GT file lists explicit empty rows, absent S1 ids are *unlabelled*
    (excluded); otherwise absent S1 ids are taken to have zero matches.
    """
    if cfg.gt_absent_policy != "auto":
        return cfg.gt_absent_policy
    has_empty = parse_gt(gt).filter(pl.col("matched").list.len() == 0).height > 0
    return "exclude" if has_empty else "no_match"


__all__ = ["DataError", "discover_files", "discover_tables", "canonical_table_name", "materialise_raw",
           "manifest", "scan", "load", "parse_gt", "gt_pairs", "valid_gt_pairs", "validate_tables",
           "resolve_gt_policy", "TRAIN_TABLES", "TEST_TABLES"]

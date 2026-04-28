import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "formal"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect experiment results from output directories.")
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=str(DEFAULT_OUTPUTS_ROOT),
        help="Root directory containing experiment output folders.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional CSV output path. Defaults to <outputs_root>/experiment_summary.csv",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional JSON output path. Defaults to <outputs_root>/experiment_summary.json",
    )
    return parser.parse_args()


def safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def collect_one(output_dir: Path):
    cfg = safe_load_json(output_dir / "config_resolved.json") or {}
    best_metrics = safe_load_json(output_dir / "best_metrics.json") or {}
    best_sweep_metrics = safe_load_json(output_dir / "best_sweep_metrics.json") or {}
    test_metrics = safe_load_json(output_dir / "test_metrics.json") or {}
    test_sweep_metrics = safe_load_json(output_dir / "test_sweep_metrics.json") or {}

    if not cfg and not best_metrics and not best_sweep_metrics and not test_metrics and not test_sweep_metrics:
        return None

    row = {
        "model_name": cfg.get("model_name"),
        "config_name": first_non_none(cfg.get("config_name"), Path(cfg.get("config_path", "")).name or None, output_dir.name),
        "output_dir": str(output_dir.resolve()),
        "seed": cfg.get("seed"),
        "best_fixed_val_f1": first_non_none(best_metrics.get("val_f1"), best_metrics.get("best_fixed_f1")),
        "best_fixed_val_epoch": first_non_none(best_metrics.get("epoch"), best_metrics.get("best_fixed_epoch")),
        "best_sweep_val_f1": first_non_none(
            best_sweep_metrics.get("best_sweep_f1"),
            best_sweep_metrics.get("val_best_threshold_f1"),
            best_sweep_metrics.get("val_f1"),
        ),
        "best_sweep_threshold": first_non_none(
            best_sweep_metrics.get("best_sweep_threshold"),
            best_sweep_metrics.get("val_best_threshold"),
        ),
        "best_sweep_epoch": first_non_none(
            best_sweep_metrics.get("best_sweep_epoch"),
            best_sweep_metrics.get("epoch"),
        ),
        "test_fixed_f1": first_non_none(test_metrics.get("test_fixed_f1"), test_metrics.get("f1")),
        "test_fixed_iou": first_non_none(test_metrics.get("test_fixed_iou"), test_metrics.get("iou")),
        "test_sweep_f1": first_non_none(test_metrics.get("test_sweep_f1"), test_sweep_metrics.get("test_sweep_f1")),
        "test_sweep_iou": first_non_none(test_metrics.get("test_sweep_iou"), test_sweep_metrics.get("test_sweep_iou")),
        "test_sweep_threshold": first_non_none(
            test_metrics.get("test_sweep_threshold"),
            test_sweep_metrics.get("test_sweep_threshold"),
        ),
    }
    return row


def collect_all(outputs_root: Path):
    if not outputs_root.exists():
        return []

    rows = []
    seen = set()
    for cfg_path in sorted(outputs_root.rglob("config_resolved.json")):
        output_dir = cfg_path.parent.resolve()
        if output_dir in seen:
            continue
        seen.add(output_dir)
        row = collect_one(output_dir)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "config_name",
        "output_dir",
        "seed",
        "best_fixed_val_f1",
        "best_fixed_val_epoch",
        "best_sweep_val_f1",
        "best_sweep_threshold",
        "best_sweep_epoch",
        "test_fixed_f1",
        "test_fixed_iou",
        "test_sweep_f1",
        "test_sweep_iou",
        "test_sweep_threshold",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else outputs_root / "experiment_summary.csv"
    out_json = Path(args.out_json).resolve() if args.out_json else outputs_root / "experiment_summary.json"

    rows = collect_all(outputs_root)
    rows.sort(key=lambda item: (item["model_name"] or "", item["config_name"] or "", item["output_dir"]))

    write_csv(out_csv, rows)
    write_json(out_json, rows)

    print(f"outputs_root: {outputs_root}")
    print(f"runs_collected: {len(rows)}")
    print(f"csv: {out_csv}")
    print(f"json: {out_json}")


if __name__ == "__main__":
    main()

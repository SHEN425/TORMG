import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-run experiment outputs (detail + grouped mean/std)."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        required=True,
        help="Root directory that contains multiple run folders with config_resolved.json.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for aggregate files. Defaults to <runs_root>/aggregates.",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="model_name",
        choices=["model_name", "config_name", "model_and_config"],
        help="Grouping key for grouped summary.",
    )
    parser.add_argument(
        "--efficiency_json",
        type=str,
        default="",
        help="Optional efficiency benchmark JSON path to export a clean summary table.",
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


def pick_metric(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def as_float_or_none(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_runs(runs_root: Path):
    rows = []
    for cfg_path in sorted(runs_root.rglob("config_resolved.json")):
        run_dir = cfg_path.parent
        cfg = safe_load_json(cfg_path) or {}
        best_metrics = safe_load_json(run_dir / "best_metrics.json") or {}
        best_sweep_metrics = safe_load_json(run_dir / "best_sweep_metrics.json") or {}
        test_metrics = safe_load_json(run_dir / "test_metrics.json") or {}
        test_sweep_metrics = safe_load_json(run_dir / "test_sweep_metrics.json") or {}

        row = {
            "run_dir": str(run_dir.resolve()),
            "run_name": run_dir.name,
            "model_name": cfg.get("model_name"),
            "config_name": pick_metric(cfg, "config_name") or Path(cfg.get("config_path", "")).name or run_dir.name,
            "seed": cfg.get("seed"),
            "device": cfg.get("device"),
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "gt_type": cfg.get("gt_type"),
            "train_ratio": cfg.get("train_ratio"),
            "val_ratio": cfg.get("val_ratio"),
            "test_ratio": cfg.get("test_ratio"),
            "best_fixed_val_f1": as_float_or_none(pick_metric(best_metrics, "val_f1", "val_chd_f1", "best_fixed_f1")),
            "best_fixed_val_iou": as_float_or_none(pick_metric(best_metrics, "val_iou", "val_chd_iou")),
            "best_fixed_val_epoch": pick_metric(best_metrics, "epoch", "best_fixed_epoch"),
            "best_sweep_val_f1": as_float_or_none(
                pick_metric(best_sweep_metrics, "best_sweep_f1", "val_best_threshold_f1", "val_f1", "val_chd_f1")
            ),
            "best_sweep_val_iou": as_float_or_none(
                pick_metric(best_sweep_metrics, "best_sweep_iou", "val_best_threshold_iou", "val_iou", "val_chd_iou")
            ),
            "best_sweep_threshold": as_float_or_none(
                pick_metric(best_sweep_metrics, "best_sweep_threshold", "val_best_threshold")
            ),
            "test_fixed_f1": as_float_or_none(pick_metric(test_metrics, "test_fixed_f1", "f1", "chd_f1")),
            "test_fixed_iou": as_float_or_none(pick_metric(test_metrics, "test_fixed_iou", "iou", "chd_iou")),
            "test_sweep_f1": as_float_or_none(pick_metric(test_metrics, "test_sweep_f1")),
            "test_sweep_iou": as_float_or_none(pick_metric(test_metrics, "test_sweep_iou")),
            "test_sweep_threshold": as_float_or_none(pick_metric(test_metrics, "test_sweep_threshold")),
        }

        if row["test_sweep_f1"] is None:
            row["test_sweep_f1"] = as_float_or_none(test_sweep_metrics.get("test_sweep_f1"))
        if row["test_sweep_iou"] is None:
            row["test_sweep_iou"] = as_float_or_none(test_sweep_metrics.get("test_sweep_iou"))
        if row["test_sweep_threshold"] is None:
            row["test_sweep_threshold"] = as_float_or_none(test_sweep_metrics.get("test_sweep_threshold"))

        rows.append(row)
    return rows


def group_key(row: dict, mode: str):
    if mode == "model_name":
        return str(row.get("model_name"))
    if mode == "config_name":
        return str(row.get("config_name"))
    return f"{row.get('model_name')}::{row.get('config_name')}"


def finite_values(rows, field):
    vals = []
    for r in rows:
        v = r.get(field)
        if isinstance(v, (int, float)) and math.isfinite(v):
            vals.append(float(v))
    return vals


def summarize_group(rows, key_name, key_value):
    out = {
        key_name: key_value,
        "n_runs": len(rows),
        "seeds": ",".join(str(r["seed"]) for r in rows if r.get("seed") is not None),
    }
    metric_fields = [
        "best_fixed_val_f1",
        "best_fixed_val_iou",
        "best_sweep_val_f1",
        "best_sweep_val_iou",
        "test_fixed_f1",
        "test_fixed_iou",
        "test_sweep_f1",
        "test_sweep_iou",
    ]
    for field in metric_fields:
        vals = finite_values(rows, field)
        out[f"{field}_n"] = len(vals)
        out[f"{field}_mean"] = mean(vals) if vals else None
        out[f"{field}_std"] = stdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else None
    return out


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_efficiency(eff_path: Path):
    payload = safe_load_json(eff_path)
    if not payload:
        return []
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []

    grouped = {}
    for row in rows:
        name = row.get("model_name")
        grouped.setdefault(name, []).append(row)

    out = []
    for name, items in grouped.items():
        lat = [float(x["latency_ms"]) for x in items if x.get("latency_ms") is not None]
        mem = [float(x["peak_gpu_memory_mb"]) for x in items if x.get("peak_gpu_memory_mb") is not None]
        flops = [float(x["flops"]) for x in items if x.get("flops") is not None]
        params = [float(x["params"]) for x in items if x.get("params") is not None]
        out.append(
            {
                "model_name": name,
                "n_rows": len(items),
                "latency_ms_mean": mean(lat) if lat else None,
                "latency_ms_std": stdev(lat) if len(lat) > 1 else 0.0 if len(lat) == 1 else None,
                "peak_gpu_memory_mb_mean": mean(mem) if mem else None,
                "flops_mean": mean(flops) if flops else None,
                "params_mean": mean(params) if params else None,
            }
        )
    out.sort(key=lambda x: (x["model_name"] or ""))
    return out


def main():
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs_root / "aggregates"

    detail_rows = collect_runs(runs_root)
    detail_rows.sort(key=lambda x: (str(x.get("model_name")), str(x.get("config_name")), str(x.get("run_name"))))

    grouped = {}
    for row in detail_rows:
        k = group_key(row, args.group_by)
        grouped.setdefault(k, []).append(row)

    grouped_rows = [summarize_group(v, args.group_by, k) for k, v in sorted(grouped.items(), key=lambda x: x[0])]

    write_csv(out_dir / "run_detail.csv", detail_rows)
    write_json(out_dir / "run_detail.json", detail_rows)
    write_csv(out_dir / "group_summary.csv", grouped_rows)
    write_json(out_dir / "group_summary.json", grouped_rows)

    if args.efficiency_json:
        eff_path = Path(args.efficiency_json).resolve()
        eff_rows = summarize_efficiency(eff_path)
        write_csv(out_dir / "efficiency_summary.csv", eff_rows)
        write_json(out_dir / "efficiency_summary.json", eff_rows)
        print(f"efficiency_rows: {len(eff_rows)}")

    print(f"runs_root: {runs_root}")
    print(f"run_count: {len(detail_rows)}")
    print(f"group_count: {len(grouped_rows)}")
    print(f"saved_dir: {out_dir}")


if __name__ == "__main__":
    main()

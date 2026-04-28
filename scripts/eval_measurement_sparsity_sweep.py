import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.dataset_cache import CacheDataset
from data_utils.dataset_rm import RMReconstructionDataset
from scripts.train_mvp import build_model, load_config, resolve_device, split_indices


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate envgrid_plus vs rm_unet under measurement sparsity sweep.")
    p.add_argument("--env_run_dir", type=str, required=True)
    p.add_argument("--rm_run_dir", type=str, required=True)
    p.add_argument("--env_checkpoint", type=str, default="best.pt")
    p.add_argument("--rm_checkpoint", type=str, default="best.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--keep_counts", type=int, nargs="+", default=[256, 128, 64, 32, 16])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--subsample_seed", type=int, default=1234)
    p.add_argument("--max_test_samples", type=int, default=0, help="0 means all test samples")
    p.add_argument("--out_prefix", type=str, default="sparsity_sweep_env_vs_rm")
    return p.parse_args()


def load_resolved_cfg(run_dir: Path) -> dict:
    config_path = run_dir / "config_resolved.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_model(cfg: dict, ckpt_path: Path, device: torch.device):
    model = build_model(cfg).to(device)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[warn] missing keys while loading {ckpt_path.name}: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys while loading {ckpt_path.name}: {unexpected}")
    model.eval()
    return model


def ensure_same_split_protocol(env_cfg: dict, rm_cfg: dict):
    keys = ["seed", "train_ratio", "val_ratio", "test_ratio"]
    mismatches = []
    for k in keys:
        if float(env_cfg.get(k, -1)) != float(rm_cfg.get(k, -1)):
            mismatches.append((k, env_cfg.get(k), rm_cfg.get(k)))
    if mismatches:
        msg = "; ".join([f"{k}: env={a} rm={b}" for k, a, b in mismatches])
        raise ValueError(f"Split protocol mismatch between env/rm runs: {msg}")


def hard_metrics(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    pred = pred.float().detach().cpu()
    gt = gt.float().detach().cpu()
    tp = float((pred * gt).sum().item())
    fp = float((pred * (1.0 - gt)).sum().item())
    fn = float(((1.0 - pred) * gt).sum().item())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": float(f1), "iou": float(iou)}


def to_device_sample(sample: dict, device: torch.device):
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device)
        else:
            out[k] = v
    return out


def subsample_measurements(meas_xy: torch.Tensor, meas_v: torch.Tensor, keep: int, rng: np.random.Generator):
    n = int(meas_xy.shape[0])
    k = min(max(int(keep), 1), n)
    if k == n:
        return meas_xy, meas_v
    idx = rng.choice(n, size=k, replace=False)
    idx = np.sort(idx)
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    return meas_xy[idx_t], meas_v[idx_t]


def compute_final_from_counts(stats: dict, eps: float = 1e-8):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return float(f1), float(iou)


def main():
    args = parse_args()

    env_run_dir = Path(args.env_run_dir).resolve()
    rm_run_dir = Path(args.rm_run_dir).resolve()
    env_cfg = load_resolved_cfg(env_run_dir)
    rm_cfg = load_resolved_cfg(rm_run_dir)

    ensure_same_split_protocol(env_cfg, rm_cfg)
    device = resolve_device(args.device)

    env_model = load_checkpoint_model(env_cfg, env_run_dir / args.env_checkpoint, device)
    rm_model = load_checkpoint_model(rm_cfg, rm_run_dir / args.rm_checkpoint, device)

    cache_ds = CacheDataset(
        cache_dir=env_cfg["cache_dir"],
        gt_type=env_cfg.get("gt_type", "soft"),
        physical_gt_dir=env_cfg.get("physical_gt_dir"),
        fallback_to_cache_gt=env_cfg.get("fallback_to_cache_gt", True),
    )
    rm_ds = RMReconstructionDataset(
        cache_dir=rm_cfg["cache_dir"],
        physical_gt_dir=rm_cfg["physical_gt_dir"],
        fallback_to_cache_gt=rm_cfg.get("fallback_to_cache_gt", False),
        normalize_rm=rm_cfg.get("normalize_rm", True),
    )
    if len(cache_ds) != len(rm_ds):
        raise RuntimeError(f"Dataset length mismatch: cache={len(cache_ds)} rm={len(rm_ds)}")

    _, _, test_indices = split_indices(
        total=len(cache_ds),
        train_ratio=float(env_cfg["train_ratio"]),
        seed=int(env_cfg["seed"]),
        val_ratio=float(env_cfg.get("val_ratio")) if env_cfg.get("val_ratio") is not None else None,
        test_ratio=float(env_cfg.get("test_ratio", 0.0)),
    )
    if args.max_test_samples and args.max_test_samples > 0:
        test_indices = test_indices[: int(args.max_test_samples)]

    env_thr = float(env_cfg.get("pred_threshold", 0.5))
    rm_thr = float(rm_cfg.get("pred_threshold", 0.5))
    rm_margin = float(rm_cfg.get("margin_scale", 10.0))
    rm_service_thr = float(rm_cfg.get("service_threshold_proxy", 112.0))
    rm_scale = 255.0 if bool(rm_cfg.get("normalize_rm", True)) else 1.0

    keep_counts = sorted(set(int(x) for x in args.keep_counts), reverse=True)

    accum = {}
    for keep in keep_counts:
        accum[keep] = {
            "env": {"tp": 0.0, "fp": 0.0, "fn": 0.0},
            "rm": {"tp": 0.0, "fp": 0.0, "fn": 0.0},
            "env_win": 0,
            "rm_win": 0,
            "ties": 0,
            "n_instances": 0,
            "sum_delta_f1": 0.0,
            "sum_delta_iou": 0.0,
        }

    with torch.no_grad():
        for repeat in range(int(args.repeats)):
            for idx in test_indices:
                cache_sample = cache_ds[idx]
                rm_sample = rm_ds[idx]

                gt = cache_sample["y_grid"].float()
                gt_hard = (gt > 0.5).float() if env_cfg.get("gt_type", "soft") == "soft" else gt

                n_meas = int(cache_sample["meas_xy"].shape[0])
                for keep in keep_counts:
                    rng_seed = int(args.subsample_seed) + repeat * 1000003 + idx * 9973 + keep * 433
                    rng = np.random.default_rng(rng_seed)

                    sub_meas_xy, sub_meas_v = subsample_measurements(
                        cache_sample["meas_xy"], cache_sample["meas_v"], keep=keep, rng=rng
                    )

                    env_in = {
                        "city": cache_sample["city"].unsqueeze(0).to(device),
                        "meas_xy": sub_meas_xy.unsqueeze(0).to(device),
                        "meas_v": sub_meas_v.unsqueeze(0).to(device),
                        "bs_xy": cache_sample["bs_xy"].unsqueeze(0).to(device),
                    }
                    env_logits = env_model(env_in["meas_xy"], env_in["meas_v"], env_in["bs_xy"], city=env_in["city"])
                    env_prob = torch.sigmoid(env_logits).squeeze(0)
                    if tuple(env_prob.shape[-2:]) != tuple(gt_hard.shape[-2:]):
                        env_prob = F.interpolate(
                            env_prob.unsqueeze(0).unsqueeze(0), size=gt_hard.shape, mode="area"
                        ).squeeze(0).squeeze(0)
                    env_pred = (env_prob > env_thr).float()

                    rm_in = {
                        "city": rm_sample["city"].unsqueeze(0).to(device),
                        "meas_xy": sub_meas_xy.unsqueeze(0).to(device),
                        "meas_v": sub_meas_v.unsqueeze(0).to(device),
                        "bs_xy": rm_sample["bs_xy"].unsqueeze(0).to(device),
                    }
                    rm_logits = rm_model(rm_in["meas_xy"], rm_in["meas_v"], rm_in["bs_xy"], city=rm_in["city"])
                    rm_pred = torch.sigmoid(rm_logits).squeeze(0)
                    city_aligned = rm_in["city"].squeeze(0)
                    if tuple(city_aligned.shape[-2:]) != tuple(rm_pred.shape[-2:]):
                        city_aligned = F.interpolate(
                            city_aligned.unsqueeze(0).unsqueeze(0), size=rm_pred.shape[-2:], mode="nearest"
                        ).squeeze(0).squeeze(0)
                    free_mask = (city_aligned < 0.5).float()
                    pred_radio = rm_pred * rm_scale
                    rm_prob = 1.0 / (1.0 + torch.exp((pred_radio - rm_service_thr) / max(rm_margin, 1e-6)))
                    rm_prob = rm_prob * free_mask
                    if tuple(rm_prob.shape[-2:]) != tuple(gt_hard.shape[-2:]):
                        rm_prob = F.interpolate(
                            rm_prob.unsqueeze(0).unsqueeze(0), size=gt_hard.shape, mode="area"
                        ).squeeze(0).squeeze(0)
                    rm_pred_hard = (rm_prob > rm_thr).float()

                    env_m = hard_metrics(env_pred, gt_hard)
                    rm_m = hard_metrics(rm_pred_hard, gt_hard)

                    a = accum[keep]
                    a["env"]["tp"] += env_m["tp"]
                    a["env"]["fp"] += env_m["fp"]
                    a["env"]["fn"] += env_m["fn"]
                    a["rm"]["tp"] += rm_m["tp"]
                    a["rm"]["fp"] += rm_m["fp"]
                    a["rm"]["fn"] += rm_m["fn"]
                    a["sum_delta_f1"] += (env_m["f1"] - rm_m["f1"])
                    a["sum_delta_iou"] += (env_m["iou"] - rm_m["iou"])
                    a["n_instances"] += 1

                    if env_m["f1"] > rm_m["f1"]:
                        a["env_win"] += 1
                    elif rm_m["f1"] > env_m["f1"]:
                        a["rm_win"] += 1
                    else:
                        a["ties"] += 1

    rows = []
    for keep in keep_counts:
        a = accum[keep]
        env_f1, env_iou = compute_final_from_counts(a["env"])
        rm_f1, rm_iou = compute_final_from_counts(a["rm"])
        n = max(int(a["n_instances"]), 1)
        rows.append(
            {
                "keep_count": int(keep),
                "n_instances": int(a["n_instances"]),
                "env_f1": env_f1,
                "rm_f1": rm_f1,
                "env_iou": env_iou,
                "rm_iou": rm_iou,
                "delta_f1_env_minus_rm": env_f1 - rm_f1,
                "delta_iou_env_minus_rm": env_iou - rm_iou,
                "env_win": int(a["env_win"]),
                "rm_win": int(a["rm_win"]),
                "ties": int(a["ties"]),
                "env_win_rate": float(a["env_win"]) / n,
                "rm_win_rate": float(a["rm_win"]) / n,
                "tie_rate": float(a["ties"]) / n,
                "mean_sample_delta_f1": float(a["sum_delta_f1"]) / n,
                "mean_sample_delta_iou": float(a["sum_delta_iou"]) / n,
            }
        )

    out_json = env_run_dir / f"{args.out_prefix}.json"
    out_csv = env_run_dir / f"{args.out_prefix}.csv"

    payload = {
        "env_run_dir": str(env_run_dir),
        "rm_run_dir": str(rm_run_dir),
        "env_checkpoint": args.env_checkpoint,
        "rm_checkpoint": args.rm_checkpoint,
        "device": str(device),
        "keep_counts": keep_counts,
        "repeats": int(args.repeats),
        "subsample_seed": int(args.subsample_seed),
        "n_test_samples": len(test_indices),
        "max_test_samples": int(args.max_test_samples),
        "rows": rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = list(rows[0].keys()) if rows else ["keep_count", "n_instances"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")
    print(json.dumps({"n_rows": len(rows), "n_test_samples": len(test_indices), "repeats": int(args.repeats)}, indent=2))


if __name__ == "__main__":
    main()

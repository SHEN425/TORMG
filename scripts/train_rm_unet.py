import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.dataset_rm import RMReconstructionDataset
from scripts.train_mvp import (
    build_model,
    build_threshold_list,
    is_better_sweep_candidate,
    limit_indices,
    load_config,
    resolve_device,
    resolve_task,
    setup_seed,
    split_indices,
)
from task_registry import get_num_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet RM-reconstruction baseline and evaluate CHD via RM->hole conversion.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_rm_unet_soft_test_smoke.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume from. Use 'last' to resume from <output_dir>/last.pt.",
    )
    return parser.parse_args()


def append_jsonl(path: Path, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, cfg: dict):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer_state = ckpt.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return ckpt


def attach_progress_cfg(loader, max_batches, print_batch_progress: bool):
    loader._tormg_progress_cfg = {
        "max_batches": max_batches,
        "print_batch_progress": print_batch_progress,
    }


def build_loaders(dataset: RMReconstructionDataset, cfg: dict):
    train_indices, val_indices, test_indices = split_indices(
        len(dataset),
        cfg["train_ratio"],
        cfg["seed"],
        val_ratio=cfg.get("val_ratio"),
        test_ratio=cfg.get("test_ratio", 0.0),
    )
    train_indices = limit_indices(train_indices, cfg.get("train_max_samples"))
    val_indices = limit_indices(val_indices, cfg.get("val_max_samples"))
    test_indices = limit_indices(test_indices, cfg.get("test_max_samples"))

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError("Train/val split produced an empty subset after applying sample-count limits.")

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = None if len(test_indices) == 0 else Subset(dataset, test_indices)

    train_gen = torch.Generator()
    train_gen.manual_seed(cfg["seed"])

    pin_memory = str(cfg["device"]).lower() in {"cuda", "auto"}

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices


def move_batch_to_device(batch: dict, device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def rm_shape_align(y_rm: torch.Tensor, city: torch.Tensor, out_hw):
    if tuple(y_rm.shape[-2:]) == tuple(out_hw):
        y = y_rm
    else:
        y = F.interpolate(y_rm.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False).squeeze(1)

    if tuple(city.shape[-2:]) == tuple(out_hw):
        city_out = city
    else:
        city_out = F.interpolate(city.unsqueeze(1), size=out_hw, mode="nearest").squeeze(1)
    return y, city_out


def rm_to_hole_maps(rm_pred: torch.Tensor, rm_gt: torch.Tensor, city: torch.Tensor, service_threshold: float, margin_scale: float, normalize_rm: bool):
    scale = 255.0 if normalize_rm else 1.0
    pred_radio = rm_pred * scale
    gt_radio = rm_gt * scale

    free_mask = (city < 0.5).float()

    margin_den = max(float(margin_scale), 1e-6)
    pred_soft = 1.0 / (1.0 + torch.exp((pred_radio - service_threshold) / margin_den))
    gt_soft = 1.0 / (1.0 + torch.exp((gt_radio - service_threshold) / margin_den))
    pred_soft = pred_soft * free_mask
    gt_soft = gt_soft * free_mask

    gt_hard = ((gt_radio < service_threshold) & (free_mask > 0.5)).float()
    return pred_soft, gt_soft, gt_hard, free_mask


def compute_stats_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou


def update_threshold_sweep(sweep: dict, pred_soft: torch.Tensor, gt_hard: torch.Tensor):
    pred_soft_cpu = pred_soft.detach().float().cpu()
    gt_hard_cpu = gt_hard.detach().float().cpu()
    for threshold, stats in sweep.items():
        pred_cpu = (pred_soft_cpu > threshold).float()
        stats["tp"] += float((pred_cpu * gt_hard_cpu).sum().item())
        stats["fp"] += float((pred_cpu * (1.0 - gt_hard_cpu)).sum().item())
        stats["fn"] += float(((1.0 - pred_cpu) * gt_hard_cpu).sum().item())
        stats["pred_pos"] += float(pred_cpu.sum().item())
        stats["pixel_count"] += pred_cpu.numel()


def init_threshold_sweep(thresholds):
    return {
        float(th): {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pred_pos": 0.0,
            "pixel_count": 0,
        }
        for th in thresholds
    }


def finalize_threshold_sweep(sweep: dict):
    results = []
    best = None
    for threshold in sorted(sweep.keys()):
        stats = sweep[threshold]
        f1, iou = compute_stats_from_counts(stats["tp"], stats["fp"], stats["fn"])
        pred_pos_ratio = stats["pred_pos"] / max(int(stats["pixel_count"]), 1)
        item = {
            "threshold": float(threshold),
            "f1": f1,
            "iou": iou,
            "pred_pos_ratio": pred_pos_ratio,
        }
        results.append(item)
        if best is None or item["f1"] > best["f1"] or (item["f1"] == best["f1"] and item["iou"] > best["iou"]):
            best = item
    return results, best


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    w = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / w


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    w = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / w


def rm_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_name: str):
    name = str(loss_name).lower()
    if name == "mse":
        return masked_mse(pred, target, mask)
    if name == "smooth_l1":
        w = mask.sum().clamp_min(1.0)
        return (F.smooth_l1_loss(pred, target, reduction="none") * mask).sum() / w
    return masked_l1(pred, target, mask)


def train_one_epoch(model, loader, optimizer, device, cfg: dict):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_items = 0

    tp = fp = fn = 0.0
    progress_cfg = getattr(loader, "_tormg_progress_cfg", None)
    max_batches = None if progress_cfg is None else progress_cfg.get("max_batches")

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        logits = model(batch["meas_xy"], batch["meas_v"], batch["bs_xy"], city=batch["city"])
        pred_rm = torch.sigmoid(logits)

        y_rm, city_aligned = rm_shape_align(batch["y_rm"], batch["city"], pred_rm.shape[-2:])
        free_mask = (city_aligned < 0.5).float()

        loss = rm_loss(pred_rm, y_rm, free_mask, cfg.get("rm_loss", "l1"))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mae = masked_l1(pred_rm.detach(), y_rm, free_mask)
        mse = masked_mse(pred_rm.detach(), y_rm, free_mask)
        rmse = torch.sqrt(mse.clamp_min(0.0))

        pred_soft, _, gt_hard, _ = rm_to_hole_maps(
            pred_rm.detach(),
            y_rm,
            city_aligned,
            service_threshold=float(cfg["service_threshold_proxy"]),
            margin_scale=float(cfg["margin_scale"]),
            normalize_rm=bool(cfg.get("normalize_rm", True)),
        )
        pred_hard = (pred_soft > float(cfg["pred_threshold"])) .float()

        tp += float((pred_hard * gt_hard).sum().item())
        fp += float((pred_hard * (1.0 - gt_hard)).sum().item())
        fn += float(((1.0 - pred_hard) * gt_hard).sum().item())

        batch_size = y_rm.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_rmse += float(rmse.item()) * batch_size
        total_items += batch_size

        if progress_cfg and progress_cfg.get("print_batch_progress", False):
            limit = progress_cfg.get("max_batches")
            suffix = f"/{limit}" if limit is not None else ""
            print(f"  train batch {batch_idx}{suffix} | loss {loss.item():.4f}")

        if max_batches is not None and batch_idx >= max_batches:
            break

    f1, iou = compute_stats_from_counts(tp, fp, fn)
    n = max(total_items, 1)
    return {
        "rm_loss": total_loss / n,
        "rm_mae": total_mae / n,
        "rm_rmse": total_rmse / n,
        "chd_f1": f1,
        "chd_iou": iou,
    }


@torch.no_grad()
def validate(model, loader, device, cfg: dict, threshold_sweep_values):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_items = 0

    tp = fp = fn = 0.0
    threshold_sweep = init_threshold_sweep(threshold_sweep_values)

    progress_cfg = getattr(loader, "_tormg_progress_cfg", None)
    max_batches = None if progress_cfg is None else progress_cfg.get("max_batches")

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        logits = model(batch["meas_xy"], batch["meas_v"], batch["bs_xy"], city=batch["city"])
        pred_rm = torch.sigmoid(logits)

        y_rm, city_aligned = rm_shape_align(batch["y_rm"], batch["city"], pred_rm.shape[-2:])
        free_mask = (city_aligned < 0.5).float()

        loss = rm_loss(pred_rm, y_rm, free_mask, cfg.get("rm_loss", "l1"))
        mae = masked_l1(pred_rm, y_rm, free_mask)
        mse = masked_mse(pred_rm, y_rm, free_mask)
        rmse = torch.sqrt(mse.clamp_min(0.0))

        pred_soft, _, gt_hard, _ = rm_to_hole_maps(
            pred_rm,
            y_rm,
            city_aligned,
            service_threshold=float(cfg["service_threshold_proxy"]),
            margin_scale=float(cfg["margin_scale"]),
            normalize_rm=bool(cfg.get("normalize_rm", True)),
        )

        pred_hard = (pred_soft > float(cfg["pred_threshold"])) .float()

        tp += float((pred_hard * gt_hard).sum().item())
        fp += float((pred_hard * (1.0 - gt_hard)).sum().item())
        fn += float(((1.0 - pred_hard) * gt_hard).sum().item())

        update_threshold_sweep(threshold_sweep, pred_soft, gt_hard)

        batch_size = y_rm.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_rmse += float(rmse.item()) * batch_size
        total_items += batch_size

        if progress_cfg and progress_cfg.get("print_batch_progress", False):
            limit = progress_cfg.get("max_batches")
            suffix = f"/{limit}" if limit is not None else ""
            print(f"  val/test batch {batch_idx}{suffix} | loss {loss.item():.4f}")

        if max_batches is not None and batch_idx >= max_batches:
            break

    f1, iou = compute_stats_from_counts(tp, fp, fn)
    n = max(total_items, 1)
    sweep_results, best_sweep = finalize_threshold_sweep(threshold_sweep)

    return {
        "rm_loss": total_loss / n,
        "rm_mae": total_mae / n,
        "rm_rmse": total_rmse / n,
        "chd_f1": f1,
        "chd_iou": iou,
        "threshold_sweep": sweep_results,
        "best_threshold": best_sweep,
    }


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    cfg["config_name"] = config_path.name
    cfg["config_path"] = str(config_path)
    cfg.setdefault("model_name", "unet_baseline")
    cfg.setdefault("gt_type", "soft")
    cfg.setdefault("fallback_to_cache_gt", False)
    cfg.setdefault("normalize_rm", True)
    cfg.setdefault("rm_loss", "l1")
    cfg.setdefault("service_threshold_proxy", 112.0)
    cfg.setdefault("margin_scale", 10.0)
    cfg.setdefault("pred_threshold", 0.5)
    cfg.setdefault("use_atrc_scorer", False)
    cfg.setdefault("meas_keep_count", 0)
    cfg.setdefault("meas_sample_mode", "random")
    cfg.setdefault("meas_sample_seed", cfg.get("seed", 0))
    cfg.setdefault("meas_sample_deterministic", True)

    if cfg["model_name"].lower() != "unet_baseline":
        raise ValueError(f"train_rm_unet.py only supports model_name='unet_baseline', got {cfg['model_name']}")

    task_name, task_id = resolve_task(cfg)
    cfg["task_name"] = task_name
    cfg["task_id"] = task_id
    cfg.setdefault("num_tasks", get_num_tasks())

    setup_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log_path = output_dir / "train_log.jsonl"

    dataset = RMReconstructionDataset(
        cache_dir=cfg["cache_dir"],
        physical_gt_dir=cfg["physical_gt_dir"],
        fallback_to_cache_gt=cfg.get("fallback_to_cache_gt", False),
        normalize_rm=cfg.get("normalize_rm", True),
        meas_keep_count=cfg.get("meas_keep_count"),
        meas_sample_mode=cfg.get("meas_sample_mode", "random"),
        meas_sample_seed=cfg.get("meas_sample_seed", cfg.get("seed", 0)),
        meas_sample_deterministic=cfg.get("meas_sample_deterministic", True),
    )

    train_loader, val_loader, test_loader, train_indices, val_indices, test_indices = build_loaders(dataset, cfg)
    attach_progress_cfg(train_loader, cfg.get("max_train_batches"), cfg.get("print_batch_progress", False))
    attach_progress_cfg(val_loader, cfg.get("max_val_batches"), cfg.get("print_batch_progress", False))
    if test_loader is not None:
        attach_progress_cfg(test_loader, cfg.get("max_test_batches"), cfg.get("print_batch_progress", False))

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    thresholds = build_threshold_list(cfg)

    best_f1 = -1.0
    best_metrics = None
    best_epoch = None
    best_sweep = None
    best_sweep_metrics = None
    best_sweep_epoch = None
    history = []
    start_epoch = 1

    resume_arg = str(args.resume).strip()
    if resume_arg:
        resume_path = output_dir / "last.pt" if resume_arg.lower() == "last" else Path(resume_arg).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        ckpt_epoch = int(ckpt.get("epoch", 0))
        ckpt_metrics = ckpt.get("metrics") or {}
        start_epoch = ckpt_epoch + 1

        history_path = output_dir / "metrics_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                loaded_history = json.load(f)
            if isinstance(loaded_history, list):
                history = loaded_history

        if ckpt_metrics:
            best_f1 = float(ckpt_metrics.get("best_fixed_f1", ckpt_metrics.get("val_chd_f1", -1.0)))
            best_epoch = ckpt_metrics.get("best_fixed_epoch", ckpt_epoch)
            if "best_sweep_threshold" in ckpt_metrics and ckpt_metrics.get("best_sweep_threshold") is not None:
                best_sweep = {
                    "threshold": float(ckpt_metrics["best_sweep_threshold"]),
                    "f1": float(ckpt_metrics.get("best_sweep_f1", -1.0)),
                    "iou": float(ckpt_metrics.get("best_sweep_iou", -1.0)),
                }
                best_sweep_epoch = ckpt_metrics.get("best_sweep_epoch", ckpt_epoch)
            best_metrics = ckpt_metrics
            best_sweep_metrics = ckpt_metrics

    print(f"config: {config_path}")
    print(f"device: {device}")
    print(f"cache_dir: {cfg['cache_dir']}")
    print(f"physical_gt_dir: {cfg['physical_gt_dir']}")
    print(f"output_dir: {cfg['output_dir']}")
    print(f"model_name: {cfg['model_name']}")
    print(f"rm_loss: {cfg['rm_loss']}")
    print(f"meas_keep_count: {cfg.get('meas_keep_count', 0)}")
    print(f"meas_sample_mode: {cfg.get('meas_sample_mode', 'random')}")
    print(f"meas_sample_seed: {cfg.get('meas_sample_seed', cfg.get('seed', 0))}")
    print(f"meas_sample_deterministic: {cfg.get('meas_sample_deterministic', True)}")
    print(f"dataset size: total={len(dataset)} train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")
    if resume_arg:
        print(f"resume_from: {resume_arg}")
        print(f"start_epoch: {start_epoch}")
    if start_epoch > int(cfg["epochs"]):
        print(f"resume checkpoint already at epoch {start_epoch - 1}, target epochs={cfg['epochs']}; nothing to train.")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics = validate(model, val_loader, device, cfg, thresholds)

        print(
            f"epoch {epoch:03d} | "
            f"train_rm_loss {train_metrics['rm_loss']:.4f} | "
            f"val_rm_loss {val_metrics['rm_loss']:.4f} | "
            f"val_chd_f1 {val_metrics['chd_f1']:.4f} | "
            f"val_chd_iou {val_metrics['chd_iou']:.4f}"
        )

        current_fixed_is_best = val_metrics["chd_f1"] > best_f1
        current_sweep_is_best = is_better_sweep_candidate(val_metrics.get("best_threshold"), best_sweep)
        current_sweep_best = val_metrics.get("best_threshold") if current_sweep_is_best else best_sweep
        current_sweep_best_epoch = epoch if current_sweep_is_best else best_sweep_epoch

        epoch_metrics = {
            "epoch": epoch,
            "train_rm_loss": train_metrics["rm_loss"],
            "train_rm_mae": train_metrics["rm_mae"],
            "train_rm_rmse": train_metrics["rm_rmse"],
            "train_chd_f1": train_metrics["chd_f1"],
            "train_chd_iou": train_metrics["chd_iou"],
            "val_rm_loss": val_metrics["rm_loss"],
            "val_rm_mae": val_metrics["rm_mae"],
            "val_rm_rmse": val_metrics["rm_rmse"],
            "val_chd_f1": val_metrics["chd_f1"],
            "val_chd_iou": val_metrics["chd_iou"],
            "pred_threshold": cfg["pred_threshold"],
            "val_best_threshold": None if val_metrics.get("best_threshold") is None else val_metrics["best_threshold"]["threshold"],
            "val_best_threshold_f1": None if val_metrics.get("best_threshold") is None else val_metrics["best_threshold"]["f1"],
            "val_best_threshold_iou": None if val_metrics.get("best_threshold") is None else val_metrics["best_threshold"]["iou"],
            "val_threshold_sweep": val_metrics.get("threshold_sweep", []),
            "best_sweep_f1": None if current_sweep_best is None else current_sweep_best["f1"],
            "best_sweep_threshold": None if current_sweep_best is None else current_sweep_best["threshold"],
            "best_sweep_iou": None if current_sweep_best is None else current_sweep_best["iou"],
            "best_sweep_epoch": current_sweep_best_epoch,
            "best_fixed_f1": max(best_f1, val_metrics["chd_f1"]),
            "best_fixed_epoch": epoch if current_fixed_is_best else best_epoch,
        }
        history.append(epoch_metrics)
        append_jsonl(log_path, epoch_metrics)

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, epoch_metrics, cfg)

        if current_fixed_is_best:
            best_f1 = val_metrics["chd_f1"]
            best_metrics = epoch_metrics
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, epoch_metrics, cfg)

        if current_sweep_is_best:
            best_sweep = dict(val_metrics["best_threshold"])
            best_sweep_metrics = epoch_metrics
            best_sweep_epoch = epoch
            save_checkpoint(output_dir / "best_sweep.pt", model, optimizer, epoch, epoch_metrics, cfg)

    with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if best_metrics is not None:
        with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)
    if best_sweep_metrics is not None:
        with open(output_dir / "best_sweep_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_sweep_metrics, f, indent=2)

    if test_loader is not None and (output_dir / "best.pt").exists():
        best_ckpt = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])

        test_metrics = validate(model, test_loader, device, cfg, thresholds)

        sweep_test_metrics = None
        sweep_test_threshold = None

        best_sweep_ckpt_path = output_dir / "best_sweep.pt"
        best_sweep_metrics_path = output_dir / "best_sweep_metrics.json"
        if best_sweep_ckpt_path.exists() and best_sweep_metrics_path.exists():
            with open(best_sweep_metrics_path, "r", encoding="utf-8") as f:
                best_sweep_meta = json.load(f)
            sweep_test_threshold = best_sweep_meta.get("best_sweep_threshold")

            if sweep_test_threshold is not None:
                best_sweep_ckpt = torch.load(best_sweep_ckpt_path, map_location=device)
                model.load_state_dict(best_sweep_ckpt["model_state"])
                sweep_test_metrics = validate(model, test_loader, device, {**cfg, "pred_threshold": float(sweep_test_threshold)}, [])

        test_metrics_out = dict(test_metrics)
        test_metrics_out["test_fixed_f1"] = test_metrics["chd_f1"]
        test_metrics_out["test_fixed_iou"] = test_metrics["chd_iou"]
        test_metrics_out["test_fixed_threshold"] = float(cfg["pred_threshold"])
        test_metrics_out["test_sweep_f1"] = None if sweep_test_metrics is None else sweep_test_metrics["chd_f1"]
        test_metrics_out["test_sweep_iou"] = None if sweep_test_metrics is None else sweep_test_metrics["chd_iou"]
        test_metrics_out["test_sweep_threshold"] = None if sweep_test_threshold is None else float(sweep_test_threshold)

        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics_out, f, indent=2)

        if sweep_test_metrics is not None and sweep_test_threshold is not None:
            sweep_test_out = dict(sweep_test_metrics)
            sweep_test_out["test_sweep_f1"] = sweep_test_metrics["chd_f1"]
            sweep_test_out["test_sweep_iou"] = sweep_test_metrics["chd_iou"]
            sweep_test_out["test_sweep_threshold"] = float(sweep_test_threshold)
            with open(output_dir / "test_sweep_metrics.json", "w", encoding="utf-8") as f:
                json.dump(sweep_test_out, f, indent=2)


if __name__ == "__main__":
    main()

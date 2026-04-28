import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.dataset_cache import CacheDataset
from models.atrc import ATRCScorer
from models.mvp_crossattn_allgrid import MVP_CrossAttn_AllGrid
from models.mvp_crossattn import MVP_CrossAttn
from models.mvp_crossattn_envgrid import MVP_CrossAttn_EnvGrid
from models.mvp_crossattn_envgrid_plus import MVP_CrossAttn_EnvGridPlus
from models.mvp_crossattn_envtokens import MVP_CrossAttn_EnvTokens
from models.mvp_crossattn_gridbias import MVP_CrossAttn_GridBias
from models.sanity_soft_baseline import SanitySoftBaseline
from models.unet_baseline import UNetBaseline
from models.fcn_baseline import FCNBaseline
from models.deeplabv3_lite_baseline import DeepLabV3LiteBaseline
from task_registry import get_num_tasks, resolve_task


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MVP coverage-hole model on cached samples.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_mvp.json"),
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["cache_dir"] = str(resolve_path(config_path.parent, cfg["cache_dir"]))
    cfg["output_dir"] = str(resolve_path(config_path.parent, cfg["output_dir"]))
    if cfg.get("physical_gt_dir"):
        cfg["physical_gt_dir"] = str(resolve_path(config_path.parent, cfg["physical_gt_dir"]))
    return cfg


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    name = device_name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False.")
    return torch.device(name)


def split_indices(total: int, train_ratio: float, seed: int, val_ratio: Optional[float] = None, test_ratio: float = 0.0):
    min_required = 3 if test_ratio > 0.0 else 2
    if total < min_required:
        raise RuntimeError(f"Need at least {min_required} cached samples for the requested split.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if val_ratio is None:
        remaining = 1.0 - train_ratio
        val_ratio = remaining - test_ratio
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {ratio_sum}")

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if test_ratio > 0.0:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        test_count = max(1, test_count)
        overflow = train_count + val_count + test_count - total
        while overflow > 0:
            if train_count >= val_count and train_count >= test_count and train_count > 1:
                train_count -= 1
            elif val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            overflow -= 1
    else:
        train_count = max(1, min(total - 1, train_count))
        val_count = total - train_count
        test_count = 0

    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count : train_count + val_count + test_count]
    return train_indices, val_indices, test_indices


def build_model(cfg: dict) -> torch.nn.Module:
    model_name = cfg.get("model_name", "mvp_crossattn").lower()
    if model_name == "mvp_crossattn":
        return MVP_CrossAttn(
            d=cfg.get("model_dim", 128),
            heads=cfg.get("model_heads", 4),
            layers=cfg.get("model_layers", 2),
            grid_size=cfg.get("grid_size", 64),
            num_tasks=cfg.get("num_tasks", get_num_tasks()),
            patch_size=cfg.get("patch_size", 9),
        )
    if model_name == "sanity_soft_baseline":
        return SanitySoftBaseline(
            grid_size=cfg.get("grid_size", 64),
            hidden_channels=cfg.get("baseline_hidden_channels", 32),
        )
    if model_name == "unet_baseline":
        return UNetBaseline(
            grid_size=cfg.get("grid_size", 64),
            base_channels=cfg.get("unet_base_channels", cfg.get("baseline_hidden_channels", 32)),
        )
    if model_name == "fcn_baseline":
        return FCNBaseline(
            grid_size=cfg.get("grid_size", 64),
            base_channels=cfg.get("fcn_base_channels", cfg.get("baseline_hidden_channels", 32)),
        )
    if model_name == "deeplabv3_lite_baseline":
        return DeepLabV3LiteBaseline(
            grid_size=cfg.get("grid_size", 64),
            base_channels=cfg.get("deeplab_lite_base_channels", cfg.get("baseline_hidden_channels", 32)),
        )
    if model_name == "mvp_crossattn_gridbias":
        return MVP_CrossAttn_GridBias(
            d=cfg.get("model_dim", 128),
            heads=cfg.get("model_heads", 4),
            layers=cfg.get("model_layers", 2),
            grid_size=cfg.get("grid_size", 64),
            num_tasks=cfg.get("num_tasks", get_num_tasks()),
            patch_size=cfg.get("patch_size", 9),
            gridbias_channels=cfg.get("gridbias_channels", 32),
        )
    if model_name == "mvp_crossattn_envtokens":
        return MVP_CrossAttn_EnvTokens(
            d=cfg.get("model_dim", 128),
            heads=cfg.get("model_heads", 4),
            layers=cfg.get("model_layers", 2),
            grid_size=cfg.get("grid_size", 64),
            num_tasks=cfg.get("num_tasks", get_num_tasks()),
            env_token_grid_size=cfg.get("env_token_grid_size", 16),
            env_channels=cfg.get("env_channels", 32),
        )
    if model_name == "mvp_crossattn_envgrid":
        return MVP_CrossAttn_EnvGrid(
            d=cfg.get("model_dim", 128),
            heads=cfg.get("model_heads", 4),
            layers=cfg.get("model_layers", 2),
            grid_size=cfg.get("grid_size", 64),
            num_tasks=cfg.get("num_tasks", get_num_tasks()),
            env_channels=cfg.get("env_channels", 32),
        )
    if model_name == "mvp_crossattn_envgrid_plus":
        return MVP_CrossAttn_EnvGridPlus(
            d=cfg.get("model_dim", 128),
            heads=cfg.get("model_heads", 4),
            layers=cfg.get("model_layers", 2),
            grid_size=cfg.get("grid_size", 64),
            num_tasks=cfg.get("num_tasks", get_num_tasks()),
            env_channels=cfg.get("env_channels", 32),
            enable_aux_rm_head=cfg.get("enable_aux_rm_head", False),
            aux_rm_detach_from_chd=cfg.get("aux_rm_detach_from_chd", False),
            enable_obstruction_bias=cfg.get("enable_obstruction_bias", False),
            obstruction_num_samples=cfg.get("obstruction_num_samples", 16),
            use_refine_head=cfg.get("use_refine_head", False),
            refine_channels=cfg.get("refine_channels", 32),
            refine_layers=cfg.get("refine_layers", 2),
            refine_scale=cfg.get("refine_scale", 1.0),
        )
    if model_name == "mvp_crossattn_allgrid":
        return MVP_CrossAttn_AllGrid(
            d=cfg.get("model_dim", 128),
            heads=cfg.get("model_heads", 4),
            layers=cfg.get("model_layers", 2),
            grid_size=cfg.get("grid_size", 64),
            num_tasks=cfg.get("num_tasks", get_num_tasks()),
            meas_grid_channels=cfg.get("meas_grid_channels", 32),
            bs_grid_channels=cfg.get("bs_grid_channels", 32),
            env_channels=cfg.get("env_channels", 32),
        )
    raise ValueError(f"Unsupported model_name: {cfg.get('model_name')}")


def compute_pos_weight(dataset: CacheDataset, indices, gt_type: str):
    pos = 0.0
    total = 0
    for idx in indices:
        y = dataset[idx]["y_grid"]
        if gt_type == "soft":
            y = (y > 0.5).float()
        pos += float(y.sum().item())
        total += y.numel()

    neg = total - pos
    weight = neg / max(pos, 1.0)
    return torch.tensor([weight], dtype=torch.float32)


def limit_indices(indices, max_samples):
    if max_samples is None:
        return indices
    max_samples = int(max_samples)
    if max_samples <= 0:
        return []
    return indices[: min(len(indices), max_samples)]


def build_loaders(dataset: CacheDataset, cfg: dict):
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

    if cfg.get("debug_reuse_train_for_val", False):
        if len(train_indices) == 0:
            raise RuntimeError("debug_reuse_train_for_val=True requires non-empty train_indices.")
        reuse_count = cfg.get("val_max_samples")
        if reuse_count is None:
            reuse_count = len(train_indices)
        reuse_count = max(1, min(len(train_indices), int(reuse_count)))
        val_indices = list(train_indices[:reuse_count])
        if cfg.get("disable_test_for_debug", False):
            test_indices = []

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


def build_task_id(batch: dict, device: torch.device, task_id_value: int):
    batch_size = batch["y_grid"].shape[0]
    return torch.full((batch_size,), int(task_id_value), dtype=torch.long, device=device)


def compute_stats_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou


def resolve_output_head(model):
    if hasattr(model, "coverage_hole_head"):
        return model.coverage_hole_head
    raise AttributeError("Could not locate the final output head on the model.")


def parse_model_outputs(model_out, has_feature_debug: bool, expect_aux: bool):
    if has_feature_debug:
        if expect_aux:
            logits, debug_info, aux_rm_logits = model_out
            return logits, debug_info, aux_rm_logits
        logits, debug_info = model_out
        return logits, debug_info, None

    if expect_aux:
        logits, aux_rm_logits = model_out
        return logits, None, aux_rm_logits

    return model_out, None, None


def compute_aux_rm_loss(aux_rm_logits: torch.Tensor, y_rm_target: torch.Tensor, loss_type: str):
    pred_rm = torch.sigmoid(aux_rm_logits)
    loss_name = str(loss_type).lower()
    if loss_name == "mse":
        return F.mse_loss(pred_rm, y_rm_target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(pred_rm, y_rm_target)
    if loss_name != "l1":
        raise ValueError(f"Unsupported aux_rm_loss_type: {loss_type}")
    return F.l1_loss(pred_rm, y_rm_target)


def resolve_aux_rm_target(batch: dict, out_hw):
    if "y_rm" not in batch:
        raise KeyError("Batch did not include y_rm target while aux RM head is enabled.")
    y_rm = batch["y_rm"].float()
    if tuple(y_rm.shape[-2:]) == tuple(out_hw):
        return y_rm
    y_rm = F.interpolate(y_rm.unsqueeze(1), size=out_hw, mode="area").squeeze(1)
    return y_rm


def build_boundary_mask_from_target(target: torch.Tensor, kernel_size: int):
    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    target_bin = (target > 0.5).float()
    dilate = F.max_pool2d(target_bin.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2)
    erode = 1.0 - F.max_pool2d((1.0 - target_bin).unsqueeze(1), kernel_size=k, stride=1, padding=k // 2)
    boundary = (dilate - erode).clamp(0.0, 1.0).squeeze(1)
    return boundary


def compute_chd_loss_with_boundary(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    use_boundary_loss: bool = False,
    boundary_kernel_size: int = 3,
    boundary_weight: float = 1.0,
    boundary_loss_mode: str = "weighted_bce",
    use_tversky_loss: bool = False,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    tversky_weight: float = 1.0,
    loss_mode: str = "bce",
):
    mode = str(loss_mode).lower()
    if mode not in {"bce", "bce_tversky"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    enable_tversky = bool(use_tversky_loss) or mode == "bce_tversky"

    if not use_boundary_loss:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        if not enable_tversky:
            return bce_loss, {
                "boundary_mask_mean": 0.0,
                "boundary_weight_mean": 1.0,
                "tversky_loss": 0.0,
            }
        prob = torch.sigmoid(logits)
        eps = 1e-8
        tp = (prob * target).sum()
        fp = (prob * (1.0 - target)).sum()
        fn = ((1.0 - prob) * target).sum()
        tversky = (tp + eps) / (tp + float(tversky_alpha) * fp + float(tversky_beta) * fn + eps)
        tversky_loss = 1.0 - tversky
        total_loss = bce_loss + float(tversky_weight) * tversky_loss
        return total_loss, {
            "boundary_mask_mean": 0.0,
            "boundary_weight_mean": 1.0,
            "tversky_loss": float(tversky_loss.item()),
        }

    boundary_mode = str(boundary_loss_mode).lower()
    if boundary_mode != "weighted_bce":
        raise ValueError(f"Unsupported boundary_loss_mode: {boundary_loss_mode}")

    base_bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    boundary = build_boundary_mask_from_target(target, kernel_size=boundary_kernel_size)
    pixel_weight = 1.0 + float(boundary_weight) * boundary
    bce_loss = (base_bce * pixel_weight).mean()
    tversky_loss_value = 0.0
    total_loss = bce_loss
    if enable_tversky:
        prob = torch.sigmoid(logits)
        eps = 1e-8
        tp = (prob * target).sum()
        fp = (prob * (1.0 - target)).sum()
        fn = ((1.0 - prob) * target).sum()
        tversky = (tp + eps) / (tp + float(tversky_alpha) * fp + float(tversky_beta) * fn + eps)
        tversky_loss = 1.0 - tversky
        total_loss = bce_loss + float(tversky_weight) * tversky_loss
        tversky_loss_value = float(tversky_loss.item())
    return total_loss, {
        "boundary_mask_mean": float(boundary.mean().item()),
        "boundary_weight_mean": float(pixel_weight.mean().item()),
        "tversky_loss": tversky_loss_value,
    }


def build_threshold_list(cfg: dict):
    if "val_thresholds" in cfg and cfg["val_thresholds"] is not None:
        return [float(x) for x in cfg["val_thresholds"]]
    start = float(cfg.get("val_threshold_sweep_start", 0.1))
    end = float(cfg.get("val_threshold_sweep_end", 0.9))
    step = float(cfg.get("val_threshold_sweep_step", 0.1))
    if step <= 0.0:
        raise ValueError(f"val_threshold_sweep_step must be positive, got {step}")
    values = []
    cur = start
    while cur <= end + 1e-8:
        values.append(round(cur, 4))
        cur += step
    return values


def init_prob_summary(threshold: float):
    return {
        "threshold": float(threshold),
        "pixel_count": 0,
        "prob_sum": 0.0,
        "prob_sq_sum": 0.0,
        "prob_min": float("inf"),
        "prob_max": float("-inf"),
        "pred_pos": 0.0,
        "gt_pos": 0.0,
    }


def init_value_summary():
    return {
        "count": 0,
        "sum": 0.0,
        "sq_sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def update_value_summary(summary: dict, value: torch.Tensor):
    value_cpu = value.detach().float().cpu()
    summary["count"] += value_cpu.numel()
    summary["sum"] += float(value_cpu.sum().item())
    summary["sq_sum"] += float((value_cpu * value_cpu).sum().item())
    summary["min"] = min(summary["min"], float(value_cpu.min().item()))
    summary["max"] = max(summary["max"], float(value_cpu.max().item()))


def finalize_value_summary(summary: dict, prefix: str):
    count = max(int(summary["count"]), 1)
    mean = summary["sum"] / count
    var = max(summary["sq_sum"] / count - mean * mean, 0.0)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": var ** 0.5,
        f"{prefix}_min": summary["min"] if summary["min"] != float("inf") else 0.0,
        f"{prefix}_max": summary["max"] if summary["max"] != float("-inf") else 0.0,
        f"{prefix}_norm": (summary["sq_sum"] / count) ** 0.5,
    }


def init_head_grad_summary():
    return {
        "weight_grad_norm_sum": 0.0,
        "bias_grad_norm_sum": 0.0,
        "steps": 0,
    }


def update_head_grad_summary(summary: dict, head_module):
    weight_grad = None if head_module.weight.grad is None else head_module.weight.grad.detach()
    bias_grad = None if head_module.bias is None or head_module.bias.grad is None else head_module.bias.grad.detach()
    weight_grad_norm = 0.0 if weight_grad is None else float(weight_grad.norm().item())
    bias_grad_norm = 0.0 if bias_grad is None else float(bias_grad.norm().item())
    summary["weight_grad_norm_sum"] += weight_grad_norm
    summary["bias_grad_norm_sum"] += bias_grad_norm
    summary["steps"] += 1


def finalize_head_summary(summary: dict, head_module):
    steps = max(int(summary["steps"]), 1)
    weight = head_module.weight.detach().float().cpu()
    bias = None if head_module.bias is None else head_module.bias.detach().float().cpu()
    bias_mean = 0.0 if bias is None else float(bias.mean().item())
    bias_value = 0.0 if bias is None else float(bias.view(-1)[0].item())
    return {
        "head_weight_norm": float(weight.norm().item()),
        "head_bias_mean": bias_mean,
        "head_bias_value": bias_value,
        "head_grad_norm": summary["weight_grad_norm_sum"] / steps,
        "head_bias_grad_norm": summary["bias_grad_norm_sum"] / steps,
    }


def init_feature_summaries(stage_names):
    return {name: init_value_summary() for name in stage_names}


def update_feature_summaries(summaries: dict, debug_info: dict):
    if not debug_info:
        return
    for name, tensor in debug_info.items():
        if name in summaries:
            update_value_summary(summaries[name], tensor)


def finalize_feature_summaries(summaries: dict):
    out = {}
    for name, summary in summaries.items():
        out[name] = finalize_value_summary(summary, name)
    return out


def format_feature_stage_stats(prefix: str, stage_stats: dict):
    return (
        f"{prefix} mean {stage_stats[f'{prefix}_mean']:.4f} | "
        f"std {stage_stats[f'{prefix}_std']:.4f} | "
        f"min {stage_stats[f'{prefix}_min']:.4f} | "
        f"max {stage_stats[f'{prefix}_max']:.4f} | "
        f"norm {stage_stats[f'{prefix}_norm']:.4f}"
    )


def make_feature_vis_map(tensor: torch.Tensor):
    x = tensor.detach().float().cpu()
    if x.ndim == 4:
        return torch.linalg.norm(x[0], dim=0).numpy()
    if x.ndim == 3:
        if x.shape[1] == 1:
            return torch.linalg.norm(x[0], dim=-1, keepdim=False).view(1, -1).numpy()
        token_norm = torch.linalg.norm(x[0], dim=-1)
        token_count = token_norm.numel()
        side = int(round(token_count ** 0.5))
        if side * side == token_count:
            return token_norm.view(side, side).numpy()
        return token_norm.view(1, token_count).numpy()
    if x.ndim == 2:
        return x.numpy()
    raise ValueError(f"Unsupported tensor ndim for feature visualization: {x.ndim}")


def save_feature_visualizations(split_name: str, epoch: int, output_dir: Path, feature_debug: dict):
    if not feature_debug:
        return []
    vis_dir = output_dir / f"{split_name}_feature_debug"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, tensor in feature_debug.items():
        try:
            vis_map = make_feature_vis_map(tensor)
        except ValueError:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
        interp = "nearest" if min(vis_map.shape) < 32 else "bilinear"
        im = ax.imshow(vis_map, cmap="viridis", interpolation=interp)
        ax.set_title(name, fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = vis_dir / f"{split_name}_epoch{epoch:03d}_{name}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out_path))
    return saved


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


def update_threshold_sweep(sweep: dict, prob: torch.Tensor, gt_eval: torch.Tensor):
    prob_cpu = prob.detach().float().cpu()
    gt_cpu = gt_eval.detach().float().cpu()
    for threshold, stats in sweep.items():
        pred_cpu = (prob_cpu > threshold).float()
        stats["tp"] += float((pred_cpu * gt_cpu).sum().item())
        stats["fp"] += float((pred_cpu * (1.0 - gt_cpu)).sum().item())
        stats["fn"] += float(((1.0 - pred_cpu) * gt_cpu).sum().item())
        stats["pred_pos"] += float(pred_cpu.sum().item())
        stats["pixel_count"] += pred_cpu.numel()


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


def is_better_sweep_candidate(candidate: Optional[dict], current_best: Optional[dict]) -> bool:
    if candidate is None:
        return False
    if current_best is None:
        return True
    if candidate["f1"] > current_best["f1"]:
        return True
    if candidate["f1"] < current_best["f1"]:
        return False
    if candidate["iou"] > current_best["iou"]:
        return True
    if candidate["iou"] < current_best["iou"]:
        return False
    return candidate["threshold"] < current_best["threshold"]


def update_prob_summary(summary: dict, prob: torch.Tensor, gt: torch.Tensor, threshold: float):
    prob_cpu = prob.detach().float().cpu()
    gt_cpu = gt.detach().float().cpu()
    pred_cpu = (prob_cpu > threshold).float()
    summary["pixel_count"] += prob_cpu.numel()
    summary["prob_sum"] += float(prob_cpu.sum().item())
    summary["prob_sq_sum"] += float((prob_cpu * prob_cpu).sum().item())
    summary["prob_min"] = min(summary["prob_min"], float(prob_cpu.min().item()))
    summary["prob_max"] = max(summary["prob_max"], float(prob_cpu.max().item()))
    summary["pred_pos"] += float(pred_cpu.sum().item())
    summary["gt_pos"] += float(gt_cpu.sum().item())


def finalize_prob_summary(summary: dict):
    pixel_count = max(int(summary["pixel_count"]), 1)
    prob_mean = summary["prob_sum"] / pixel_count
    prob_var = max(summary["prob_sq_sum"] / pixel_count - prob_mean * prob_mean, 0.0)
    return {
        "threshold": summary["threshold"],
        "pred_pos_ratio": summary["pred_pos"] / pixel_count,
        "gt_pos_ratio": summary["gt_pos"] / pixel_count,
        "prob_mean": prob_mean,
        "prob_std": prob_var ** 0.5,
        "prob_min": summary["prob_min"] if summary["prob_min"] != float("inf") else 0.0,
        "prob_max": summary["prob_max"] if summary["prob_max"] != float("-inf") else 0.0,
    }


def upsample_grid_for_vis(grid: torch.Tensor, out_hw):
    x = grid.float().unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=out_hw, mode="nearest")
    return x.squeeze(0).squeeze(0)


def save_split_examples(split_name: str, examples: list, output_dir: Path, threshold: float):
    if not examples:
        return []
    vis_dir = output_dir / f"{split_name}_examples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for ex in examples:
        city = ex["city"].float().cpu()
        meas_xy = ex["meas_xy"].float().cpu()
        bs_xy = ex["bs_xy"].float().cpu()
        gt = ex["gt"].float().cpu()
        prob = ex["prob"].float().cpu()
        pred = ex["pred"].float().cpu()

        h, w = city.shape
        gt_vis = upsample_grid_for_vis(gt, (h, w)).numpy()
        prob_vis = upsample_grid_for_vis(prob, (h, w)).numpy()
        pred_vis = upsample_grid_for_vis(pred, (h, w)).numpy()
        bg = np.where(city.numpy() > 0.5, 0.80, 1.00)
        meas_xy_px = meas_xy.clone()
        meas_xy_px[:, 0] = meas_xy_px[:, 0] * (w - 1)
        meas_xy_px[:, 1] = meas_xy_px[:, 1] * (h - 1)
        bs_xy_px = bs_xy.clone()
        bs_xy_px[:, 0] = bs_xy_px[:, 0] * (w - 1)
        bs_xy_px[:, 1] = bs_xy_px[:, 1] * (h - 1)

        fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.4))
        titles = [
            "City mask",
            "Sparse measurements",
            "Soft GT",
            "Prediction prob",
            f"Pred mask (>{threshold:.2f})",
        ]

        axes[0].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[1].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[1].scatter(
            meas_xy_px[:, 0].numpy(),
            meas_xy_px[:, 1].numpy(),
            s=8,
            c="#2b6cb0",
            alpha=0.85,
            linewidths=0.15,
            edgecolors="white",
        )
        axes[1].scatter(
            bs_xy_px[:, 0].numpy(),
            bs_xy_px[:, 1].numpy(),
            s=60,
            c="#d1495b",
            marker="^",
            edgecolors="white",
            linewidths=0.6,
        )
        axes[2].imshow(gt_vis, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[3].imshow(prob_vis, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[4].imshow(pred_vis, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout()
        out_path = vis_dir / f"{split_name}_epoch{ex['epoch']:03d}_idx{ex['dataset_index']:04d}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(out_path))

    return saved_paths


def train_one_epoch(
    model,
    loader,
    optimizer,
    pos_weight,
    device,
    task_id_value,
    threshold: float,
    enable_aux_rm_head: bool = False,
    aux_rm_loss_weight: float = 0.0,
    aux_rm_loss_type: str = "l1",
    aux_rm_require_target: bool = True,
    use_boundary_loss: bool = False,
    boundary_kernel_size: int = 3,
    boundary_weight: float = 1.0,
    boundary_loss_mode: str = "weighted_bce",
    use_tversky_loss: bool = False,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    tversky_weight: float = 1.0,
    loss_mode: str = "bce",
):
    model.train()
    total_loss = 0.0
    total_chd_loss = 0.0
    total_aux_rm_loss = 0.0
    total_boundary_mask_mean = 0.0
    total_boundary_weight_mean = 0.0
    total_tversky_loss = 0.0
    total_items = 0
    prob_summary = init_prob_summary(threshold)
    logit_summary = init_value_summary()
    head_grad_summary = init_head_grad_summary()
    output_head = resolve_output_head(model)
    feature_stage_names = getattr(model, "_debug_stage_names", [])
    feature_summaries = init_feature_summaries(feature_stage_names)
    progress_cfg = getattr(loader, "_tormg_progress_cfg", None)
    max_batches = None if progress_cfg is None else progress_cfg.get("max_batches")

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        task_id = build_task_id(batch, device, task_id_value)
        batch_start = time.perf_counter()

        model_kwargs = {
            "task_id": task_id,
            "city": batch["city"],
            "return_debug": bool(feature_stage_names),
        }
        if enable_aux_rm_head and hasattr(model, "enable_aux_rm_head"):
            model_kwargs["return_aux"] = True

        model_out = model(
            batch["meas_xy"],
            batch["meas_v"],
            batch["bs_xy"],
            **model_kwargs,
        )
        logits, debug_info, aux_rm_logits = parse_model_outputs(
            model_out,
            has_feature_debug=bool(feature_stage_names),
            expect_aux=enable_aux_rm_head,
        )
        if debug_info is not None:
            update_feature_summaries(feature_summaries, debug_info)

        chd_loss, boundary_stats = compute_chd_loss_with_boundary(
            logits,
            batch["y_grid"],
            pos_weight=pos_weight,
            use_boundary_loss=use_boundary_loss,
            boundary_kernel_size=boundary_kernel_size,
            boundary_weight=boundary_weight,
            boundary_loss_mode=boundary_loss_mode,
            use_tversky_loss=use_tversky_loss,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_weight=tversky_weight,
            loss_mode=loss_mode,
        )
        aux_rm_loss = torch.zeros((), device=logits.device)
        if enable_aux_rm_head:
            if aux_rm_logits is None:
                raise RuntimeError("Aux RM head was enabled, but model did not return aux_rm_logits.")
            y_rm_target = resolve_aux_rm_target(batch, out_hw=aux_rm_logits.shape[-2:])
            if y_rm_target is None and aux_rm_require_target:
                raise RuntimeError("Aux RM target is required but missing.")
            if y_rm_target is not None:
                aux_rm_loss = compute_aux_rm_loss(aux_rm_logits, y_rm_target, aux_rm_loss_type)

        loss = chd_loss + float(aux_rm_loss_weight) * aux_rm_loss
        prob = torch.sigmoid(logits)
        update_value_summary(logit_summary, logits)

        optimizer.zero_grad()
        loss.backward()
        update_head_grad_summary(head_grad_summary, output_head)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = batch["y_grid"].shape[0]
        total_loss += loss.item() * batch_size
        total_chd_loss += chd_loss.item() * batch_size
        total_aux_rm_loss += aux_rm_loss.item() * batch_size
        total_boundary_mask_mean += boundary_stats["boundary_mask_mean"] * batch_size
        total_boundary_weight_mean += boundary_stats["boundary_weight_mean"] * batch_size
        total_tversky_loss += boundary_stats["tversky_loss"] * batch_size
        total_items += batch_size
        update_prob_summary(prob_summary, prob, batch["y_grid"], threshold)

        if progress_cfg and progress_cfg.get("print_batch_progress", False):
            elapsed = time.perf_counter() - batch_start
            limit = progress_cfg.get("max_batches")
            suffix = f"/{limit}" if limit is not None else ""
            print(
                f"  train batch {batch_idx}{suffix} | total {loss.item():.4f} | "
                f"chd {chd_loss.item():.4f} | aux {aux_rm_loss.item():.4f} | {elapsed:.2f}s"
            )

        if max_batches is not None and batch_idx >= max_batches:
            break

    metrics = {
        "loss": total_loss / max(total_items, 1),
        "chd_loss": total_chd_loss / max(total_items, 1),
        "aux_rm_loss": total_aux_rm_loss / max(total_items, 1),
        "boundary_mask_mean": total_boundary_mask_mean / max(total_items, 1),
        "boundary_weight_mean": total_boundary_weight_mean / max(total_items, 1),
        "tversky_loss": total_tversky_loss / max(total_items, 1),
    }
    metrics.update(finalize_prob_summary(prob_summary))
    metrics.update(finalize_value_summary(logit_summary, "logit"))
    metrics.update(finalize_head_summary(head_grad_summary, output_head))
    metrics["feature_stats"] = finalize_feature_summaries(feature_summaries)
    return metrics


@torch.no_grad()
def validate(
    model,
    loader,
    pos_weight,
    device,
    task_id_value,
    gt_type: str,
    threshold: float,
    atrc_scorer=None,
    max_visual_examples: int = 0,
    threshold_sweep_values=None,
    enable_aux_rm_head: bool = False,
    aux_rm_loss_weight: float = 0.0,
    aux_rm_loss_type: str = "l1",
    aux_rm_require_target: bool = True,
    use_boundary_loss: bool = False,
    boundary_kernel_size: int = 3,
    boundary_weight: float = 1.0,
    boundary_loss_mode: str = "weighted_bce",
    use_tversky_loss: bool = False,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    tversky_weight: float = 1.0,
    loss_mode: str = "bce",
):
    model.eval()
    total_loss = 0.0
    total_chd_loss = 0.0
    total_aux_rm_loss = 0.0
    total_boundary_mask_mean = 0.0
    total_boundary_weight_mean = 0.0
    total_tversky_loss = 0.0
    total_items = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    example = None
    visual_examples = []
    progress_cfg = getattr(loader, "_tormg_progress_cfg", None)
    max_batches = None if progress_cfg is None else progress_cfg.get("max_batches")
    prob_summary = init_prob_summary(threshold)
    logit_summary = init_value_summary()
    threshold_sweep = init_threshold_sweep(threshold_sweep_values or [])
    feature_stage_names = getattr(model, "_debug_stage_names", [])
    feature_summaries = init_feature_summaries(feature_stage_names)
    feature_example = None

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        task_id = build_task_id(batch, device, task_id_value)
        batch_start = time.perf_counter()

        model_kwargs = {
            "task_id": task_id,
            "city": batch["city"],
            "return_debug": bool(feature_stage_names),
        }
        if enable_aux_rm_head and hasattr(model, "enable_aux_rm_head"):
            model_kwargs["return_aux"] = True

        model_out = model(
            batch["meas_xy"],
            batch["meas_v"],
            batch["bs_xy"],
            **model_kwargs,
        )
        logits, debug_info, aux_rm_logits = parse_model_outputs(
            model_out,
            has_feature_debug=bool(feature_stage_names),
            expect_aux=enable_aux_rm_head,
        )
        if debug_info is not None:
            update_feature_summaries(feature_summaries, debug_info)
            if feature_example is None:
                feature_example = {name: tensor[:1].detach().cpu() for name, tensor in debug_info.items()}

        chd_loss, boundary_stats = compute_chd_loss_with_boundary(
            logits,
            batch["y_grid"],
            pos_weight=pos_weight,
            use_boundary_loss=use_boundary_loss,
            boundary_kernel_size=boundary_kernel_size,
            boundary_weight=boundary_weight,
            boundary_loss_mode=boundary_loss_mode,
            use_tversky_loss=use_tversky_loss,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_weight=tversky_weight,
            loss_mode=loss_mode,
        )
        aux_rm_loss = torch.zeros((), device=logits.device)
        y_rm_target = None
        if enable_aux_rm_head:
            if aux_rm_logits is None:
                raise RuntimeError("Aux RM head was enabled, but model did not return aux_rm_logits.")
            y_rm_target = resolve_aux_rm_target(batch, out_hw=aux_rm_logits.shape[-2:])
            if y_rm_target is None and aux_rm_require_target:
                raise RuntimeError("Aux RM target is required but missing.")
            if y_rm_target is not None:
                aux_rm_loss = compute_aux_rm_loss(aux_rm_logits, y_rm_target, aux_rm_loss_type)

        loss = chd_loss + float(aux_rm_loss_weight) * aux_rm_loss
        update_value_summary(logit_summary, logits)

        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float()
        gt = batch["y_grid"]
        gt_eval = (gt > 0.5).float() if gt_type == "soft" else gt
        update_prob_summary(prob_summary, prob, gt_eval, threshold)
        if threshold_sweep:
            update_threshold_sweep(threshold_sweep, prob, gt_eval)

        tp += float((pred * gt_eval).sum().item())
        fp += float((pred * (1.0 - gt_eval)).sum().item())
        fn += float(((1.0 - pred) * gt_eval).sum().item())

        batch_size = gt.shape[0]
        total_loss += loss.item() * batch_size
        total_chd_loss += chd_loss.item() * batch_size
        total_aux_rm_loss += aux_rm_loss.item() * batch_size
        total_boundary_mask_mean += boundary_stats["boundary_mask_mean"] * batch_size
        total_boundary_weight_mean += boundary_stats["boundary_weight_mean"] * batch_size
        total_tversky_loss += boundary_stats["tversky_loss"] * batch_size
        total_items += batch_size

        if example is None:
            example = {
                "gt": gt[0].detach().cpu(),
                "prob": prob[0].detach().cpu(),
                "pred": pred[0].detach().cpu(),
            }
            if enable_aux_rm_head and aux_rm_logits is not None:
                example["aux_rm_prob"] = torch.sigmoid(aux_rm_logits[0]).detach().cpu()
                if y_rm_target is not None:
                    example["aux_rm_target"] = y_rm_target[0].detach().cpu()
            if atrc_scorer is not None:
                atrc_out = atrc_scorer(batch["city"], batch["meas_xy"], batch["bs_xy"])
                example["atrc"] = {
                    "measurement_density": atrc_out["measurement_density"][0].detach().cpu(),
                    "bs_density": atrc_out["bs_density"][0].detach().cpu(),
                    "urban_complexity": atrc_out["urban_complexity"][0].detach().cpu(),
                    "importance": atrc_out["importance"][0].detach().cpu(),
                    "refinement_level": atrc_out["refinement_level"][0].detach().cpu(),
                    "region_xy": atrc_out["region_xy"].detach().cpu(),
                }

        if len(visual_examples) < max_visual_examples:
            for item_idx in range(gt.shape[0]):
                if len(visual_examples) >= max_visual_examples:
                    break
                vis_item = {
                    "epoch": 0,
                    "dataset_index": batch_idx * 1000 + item_idx,
                    "city": batch["city"][item_idx].detach().cpu(),
                    "meas_xy": batch["meas_xy"][item_idx].detach().cpu(),
                    "bs_xy": batch["bs_xy"][item_idx].detach().cpu(),
                    "gt": gt[item_idx].detach().cpu(),
                    "prob": prob[item_idx].detach().cpu(),
                    "pred": pred[item_idx].detach().cpu(),
                }
                if enable_aux_rm_head and aux_rm_logits is not None:
                    vis_item["aux_rm_prob"] = torch.sigmoid(aux_rm_logits[item_idx]).detach().cpu()
                visual_examples.append(vis_item)

        if progress_cfg and progress_cfg.get("print_batch_progress", False):
            elapsed = time.perf_counter() - batch_start
            limit = progress_cfg.get("max_batches")
            suffix = f"/{limit}" if limit is not None else ""
            print(
                f"  val batch {batch_idx}{suffix} | total {loss.item():.4f} | "
                f"chd {chd_loss.item():.4f} | aux {aux_rm_loss.item():.4f} | {elapsed:.2f}s"
            )

        if max_batches is not None and batch_idx >= max_batches:
            break

    f1, iou = compute_stats_from_counts(tp, fp, fn)
    metrics = {
        "loss": total_loss / max(total_items, 1),
        "chd_loss": total_chd_loss / max(total_items, 1),
        "aux_rm_loss": total_aux_rm_loss / max(total_items, 1),
        "boundary_mask_mean": total_boundary_mask_mean / max(total_items, 1),
        "boundary_weight_mean": total_boundary_weight_mean / max(total_items, 1),
        "tversky_loss": total_tversky_loss / max(total_items, 1),
        "f1": f1,
        "iou": iou,
    }
    metrics.update(finalize_prob_summary(prob_summary))
    metrics.update(finalize_value_summary(logit_summary, "logit"))
    sweep_results, best_sweep = finalize_threshold_sweep(threshold_sweep)
    metrics["threshold_sweep"] = sweep_results
    metrics["best_threshold"] = best_sweep
    metrics["feature_stats"] = finalize_feature_summaries(feature_summaries)
    return metrics, example, visual_examples, feature_example



def attach_progress_cfg(loader, max_batches, print_batch_progress: bool):
    loader._tormg_progress_cfg = {
        "max_batches": max_batches,
        "print_batch_progress": print_batch_progress,
    }


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


def save_prediction_artifact(path: Path, example: dict, epoch: int, metrics: dict):
    artifact = dict(example)
    artifact["epoch"] = epoch
    artifact["metrics"] = metrics
    torch.save(artifact, path)


def append_jsonl(path: Path, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")


def save_training_curves(output_dir: Path, history: list):
    if not history:
        return

    epochs = [item["epoch"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    axes[0, 0].plot(epochs, [item["train_loss"] for item in history], label="Train", color="#2b6cb0", linewidth=1.8)
    axes[0, 0].plot(epochs, [item["val_loss"] for item in history], label="Val", color="#d1495b", linewidth=1.8)
    axes[0, 0].set_title("Loss", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(epochs, [item["val_f1"] for item in history], label="Val F1", color="#2f855a", linewidth=1.8)
    axes[0, 1].plot(epochs, [item["val_iou"] for item in history], label="Val IoU", color="#805ad5", linewidth=1.8)
    axes[0, 1].set_title("Validation Metrics", fontsize=10)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(epochs, [item["val_prob_mean"] for item in history], label="Prob mean", color="#718096", linewidth=1.8)
    axes[1, 0].plot(epochs, [item["val_prob_std"] for item in history], label="Prob std", color="#dd6b20", linewidth=1.8)
    axes[1, 0].set_title("Validation Probability", fontsize=10)
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(epochs, [item["val_logit_mean"] for item in history], label="Logit mean", color="#4a5568", linewidth=1.8)
    axes[1, 1].plot(epochs, [item["val_logit_std"] for item in history], label="Logit std", color="#c05621", linewidth=1.8)
    axes[1, 1].set_title("Validation Logits", fontsize=10)
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.25)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    cfg["config_name"] = config_path.name
    cfg["config_path"] = str(config_path)
    cfg.setdefault("gt_type", "soft")
    cfg.setdefault("fallback_to_cache_gt", True)
    cfg.setdefault("pred_threshold", 0.5)
    cfg.setdefault("num_visual_examples", 4)
    cfg.setdefault("use_pos_weight", True)
    cfg.setdefault("model_name", "mvp_crossattn")
    cfg.setdefault("collect_feature_debug", False)
    cfg.setdefault("enable_aux_rm_head", False)
    cfg.setdefault("aux_rm_loss_weight", 0.2)
    cfg.setdefault("aux_rm_loss_type", "l1")
    cfg.setdefault("aux_rm_detach_from_chd", False)
    cfg.setdefault("aux_rm_require_target", True)
    cfg.setdefault("aux_rm_normalize_target", True)
    cfg.setdefault("enable_obstruction_bias", False)
    cfg.setdefault("obstruction_num_samples", 16)
    cfg.setdefault("use_refine_head", False)
    cfg.setdefault("refine_channels", 32)
    cfg.setdefault("refine_layers", 2)
    cfg.setdefault("refine_scale", 1.0)
    cfg.setdefault("meas_keep_count", 0)
    cfg.setdefault("meas_sample_mode", "random")
    cfg.setdefault("meas_sample_seed", cfg.get("seed", 0))
    cfg.setdefault("meas_sample_deterministic", True)
    cfg.setdefault("use_boundary_loss", False)
    cfg.setdefault("boundary_kernel_size", 3)
    cfg.setdefault("boundary_weight", 1.0)
    cfg.setdefault("boundary_loss_mode", "weighted_bce")
    cfg.setdefault("use_tversky_loss", False)
    cfg.setdefault("tversky_alpha", 0.5)
    cfg.setdefault("tversky_beta", 0.5)
    cfg.setdefault("tversky_weight", 1.0)
    cfg.setdefault("loss_mode", "bce")
    task_name, task_id_value = resolve_task(cfg)
    cfg["task_name"] = task_name
    cfg["task_id"] = task_id_value
    cfg.setdefault("num_tasks", get_num_tasks())

    if cfg.get("enable_aux_rm_head", False) and cfg.get("model_name", "").lower() != "mvp_crossattn_envgrid_plus":
        raise ValueError("enable_aux_rm_head currently supports only mvp_crossattn_envgrid_plus.")

    setup_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log_path = output_dir / "train_log.jsonl"

    dataset = CacheDataset(
        cfg["cache_dir"],
        gt_type=cfg["gt_type"],
        physical_gt_dir=cfg.get("physical_gt_dir"),
        fallback_to_cache_gt=cfg.get("fallback_to_cache_gt", True),
        return_rm_target=cfg.get("enable_aux_rm_head", False),
        normalize_rm_target=cfg.get("aux_rm_normalize_target", True),
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
    atrc_scorer = None
    if cfg.get("use_atrc_scorer", False):
        atrc_scorer = ATRCScorer(
            region_grid_size=cfg.get("atrc_region_grid_size", 8),
            num_levels=cfg.get("atrc_num_levels", 3),
            measurement_weight=cfg.get("atrc_measurement_weight", 1.0),
            bs_weight=cfg.get("atrc_bs_weight", 1.0),
            urban_weight=cfg.get("atrc_urban_weight", 1.0),
        ).to(device)

    model = build_model(cfg).to(device)
    if cfg.get("collect_feature_debug", False):
        if cfg["model_name"].lower() == "mvp_crossattn":
            model._debug_stage_names = ["meas_tok", "src_tok", "grid_init", "grid_after_block0", "grid_out", "task_feat"]
        elif cfg["model_name"].lower() == "mvp_crossattn_gridbias":
            model._debug_stage_names = ["meas_tok", "src_tok", "grid_bias", "grid_init", "grid_after_block0", "grid_out", "task_feat", "grid_inputs"]
        elif cfg["model_name"].lower() == "mvp_crossattn_envtokens":
            model._debug_stage_names = ["meas_tok", "bs_tok", "env_tok", "src_tok", "env_feat_map", "grid_init", "grid_after_block0", "grid_out", "task_feat"]
        elif cfg["model_name"].lower() == "mvp_crossattn_envgrid":
            model._debug_stage_names = ["meas_tok", "bs_tok", "src_tok", "env_feat_map", "env_grid", "grid_init", "grid_after_block0", "grid_out", "task_feat"]
        elif cfg["model_name"].lower() == "mvp_crossattn_envgrid_plus":
            model._debug_stage_names = ["meas_tok", "bs_tok", "src_tok", "env_feat_map", "env_grid", "env_init_scale", "env_block0_scale", "grid_init", "grid_after_block0", "grid_out", "task_feat"]
            if cfg.get("enable_obstruction_bias", False):
                model._debug_stage_names.append("obstruction_ratio")
        elif cfg["model_name"].lower() == "mvp_crossattn_allgrid":
            model._debug_stage_names = ["meas_tok", "bs_tok", "src_tok", "meas_grid", "bs_grid", "env_grid", "grounded_grid", "meas_scale", "bs_scale", "env_scale", "block0_scale", "grid_init", "grid_after_block0", "grid_out", "task_feat"]
        elif cfg["model_name"].lower() == "sanity_soft_baseline":
            model._debug_stage_names = ["input_grid", "feat0", "feat1", "pre_head"]
        elif cfg["model_name"].lower() == "unet_baseline":
            model._debug_stage_names = ["input_grid", "enc1", "enc2", "bottleneck", "pre_head"]
        else:
            model._debug_stage_names = []
    else:
        model._debug_stage_names = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    pos_weight = None
    if cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weight(dataset, train_indices, cfg["gt_type"]).to(device)
    grid_size = cfg.get("grid_size", 64)
    token_count = None
    meas_tokens_for_logging = int(cfg.get("meas_keep_count", 0)) if int(cfg.get("meas_keep_count", 0)) > 0 else 256
    if cfg["model_name"].lower() in {"mvp_crossattn", "mvp_crossattn_gridbias"}:
        token_count = grid_size * grid_size + meas_tokens_for_logging + 1
    elif cfg["model_name"].lower() == "mvp_crossattn_envtokens":
        env_token_grid_size = cfg.get("env_token_grid_size", 16)
        token_count = grid_size * grid_size + meas_tokens_for_logging + 1 + env_token_grid_size * env_token_grid_size
    elif cfg["model_name"].lower() == "mvp_crossattn_envgrid":
        token_count = grid_size * grid_size + meas_tokens_for_logging + 1
    elif cfg["model_name"].lower() == "mvp_crossattn_envgrid_plus":
        token_count = grid_size * grid_size + meas_tokens_for_logging + 1
    elif cfg["model_name"].lower() == "mvp_crossattn_allgrid":
        token_count = grid_size * grid_size + meas_tokens_for_logging + 1
    threshold_sweep_values = build_threshold_list(cfg)

    best_f1 = -1.0
    best_metrics = None
    best_epoch = None
    best_sweep = None
    best_sweep_metrics = None
    best_sweep_epoch = None
    history = []

    print(f"config: {config_path}")
    print(f"device: {device}")
    print(f"cache_dir: {cfg['cache_dir']}")
    print(f"output_dir: {output_dir}")
    print(f"model_name: {cfg['model_name']}")
    print(f"task: {task_name} (id={task_id_value})")
    print(f"gt_type: {cfg['gt_type']}")
    print(f"physical_gt_dir: {cfg.get('physical_gt_dir')}")
    print(f"fallback_to_cache_gt: {cfg.get('fallback_to_cache_gt', True)}")
    print(f"pred_threshold: {cfg['pred_threshold']}")
    print(f"use_atrc_scorer: {cfg.get('use_atrc_scorer', False)}")
    print(f"enable_aux_rm_head: {cfg.get('enable_aux_rm_head', False)}")
    print(f"enable_obstruction_bias: {cfg.get('enable_obstruction_bias', False)}")
    print(f"use_refine_head: {cfg.get('use_refine_head', False)}")
    if cfg.get("use_refine_head", False):
        print(f"refine_channels: {cfg.get('refine_channels', 32)}")
        print(f"refine_layers: {cfg.get('refine_layers', 2)}")
        print(f"refine_scale: {cfg.get('refine_scale', 1.0)}")
    print(f"meas_keep_count: {cfg.get('meas_keep_count', 0)}")
    print(f"meas_sample_mode: {cfg.get('meas_sample_mode', 'random')}")
    print(f"meas_sample_seed: {cfg.get('meas_sample_seed', cfg.get('seed', 0))}")
    print(f"meas_sample_deterministic: {cfg.get('meas_sample_deterministic', True)}")
    print(f"use_boundary_loss: {cfg.get('use_boundary_loss', False)}")
    if cfg.get("use_boundary_loss", False):
        print(f"boundary_kernel_size: {cfg.get('boundary_kernel_size', 3)}")
        print(f"boundary_weight: {cfg.get('boundary_weight', 1.0)}")
        print(f"boundary_loss_mode: {cfg.get('boundary_loss_mode', 'weighted_bce')}")
    print(f"use_tversky_loss: {cfg.get('use_tversky_loss', False)}")
    print(f"loss_mode: {cfg.get('loss_mode', 'bce')}")
    if cfg.get("use_tversky_loss", False) or str(cfg.get("loss_mode", "bce")).lower() == "bce_tversky":
        print(f"tversky_alpha: {cfg.get('tversky_alpha', 0.5)}")
        print(f"tversky_beta: {cfg.get('tversky_beta', 0.5)}")
        print(f"tversky_weight: {cfg.get('tversky_weight', 1.0)}")
    if cfg.get("enable_obstruction_bias", False):
        print(f"obstruction_num_samples: {cfg.get('obstruction_num_samples', 16)}")
    if cfg.get("enable_aux_rm_head", False):
        print(f"aux_rm_loss_weight: {cfg.get('aux_rm_loss_weight', 0.2)}")
        print(f"aux_rm_loss_type: {cfg.get('aux_rm_loss_type', 'l1')}")
        print(f"aux_rm_detach_from_chd: {cfg.get('aux_rm_detach_from_chd', False)}")
    print(
        f"dataset size: total={len(dataset)} train={len(train_indices)} "
        f"val={len(val_indices)} test={len(test_indices)}"
    )
    if token_count is not None:
        if cfg["model_name"].lower() == "mvp_crossattn_envtokens":
            env_token_grid_size = cfg.get("env_token_grid_size", 16)
            print(
                f"sequence tokens per sample: {token_count} "
                f"(grid={grid_size * grid_size}, meas={meas_tokens_for_logging}, bs=1, env={env_token_grid_size * env_token_grid_size})"
            )
        elif cfg["model_name"].lower() == "mvp_crossattn_envgrid":
            print(f"sequence tokens per sample: {token_count} (grid={grid_size * grid_size}, meas={meas_tokens_for_logging}, bs=1; env anchored to grid)")
        elif cfg["model_name"].lower() == "mvp_crossattn_envgrid_plus":
            print(f"sequence tokens per sample: {token_count} (grid={grid_size * grid_size}, meas={meas_tokens_for_logging}, bs=1; env anchored to grid + early reinjection)")
        elif cfg["model_name"].lower() == "mvp_crossattn_allgrid":
            print(f"sequence tokens per sample: {token_count} (grid={grid_size * grid_size}, meas={meas_tokens_for_logging}, bs=1; all branches grounded to grid)")
        else:
            print(f"sequence tokens per sample: {token_count} (grid={grid_size * grid_size}, meas={meas_tokens_for_logging}, bs=1)")
    print(f"use_pos_weight: {cfg.get('use_pos_weight', True)}")
    print(f"pos_weight: {pos_weight.item():.4f}" if pos_weight is not None else "pos_weight: disabled")
    print(f"max_train_batches: {cfg.get('max_train_batches')}")
    print(f"max_val_batches: {cfg.get('max_val_batches')}")
    print(f"max_test_batches: {cfg.get('max_test_batches')}")
    print(f"val_threshold_sweep: {threshold_sweep_values}")

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            pos_weight,
            device,
            task_id_value,
            cfg["pred_threshold"],
            enable_aux_rm_head=cfg.get("enable_aux_rm_head", False),
            aux_rm_loss_weight=cfg.get("aux_rm_loss_weight", 0.2),
            aux_rm_loss_type=cfg.get("aux_rm_loss_type", "l1"),
            aux_rm_require_target=cfg.get("aux_rm_require_target", True),
            use_boundary_loss=cfg.get("use_boundary_loss", False),
            boundary_kernel_size=cfg.get("boundary_kernel_size", 3),
            boundary_weight=cfg.get("boundary_weight", 1.0),
            boundary_loss_mode=cfg.get("boundary_loss_mode", "weighted_bce"),
            use_tversky_loss=cfg.get("use_tversky_loss", False),
            tversky_alpha=cfg.get("tversky_alpha", 0.5),
            tversky_beta=cfg.get("tversky_beta", 0.5),
            tversky_weight=cfg.get("tversky_weight", 1.0),
            loss_mode=cfg.get("loss_mode", "bce"),
        )
        val_metrics, example, val_visual_examples, val_feature_example = validate(
            model,
            val_loader,
            pos_weight,
            device,
            task_id_value,
            cfg["gt_type"],
            cfg["pred_threshold"],
            atrc_scorer=atrc_scorer,
            max_visual_examples=cfg.get("num_visual_examples", 4),
            threshold_sweep_values=threshold_sweep_values,
            enable_aux_rm_head=cfg.get("enable_aux_rm_head", False),
            aux_rm_loss_weight=cfg.get("aux_rm_loss_weight", 0.2),
            aux_rm_loss_type=cfg.get("aux_rm_loss_type", "l1"),
            aux_rm_require_target=cfg.get("aux_rm_require_target", True),
            use_boundary_loss=cfg.get("use_boundary_loss", False),
            boundary_kernel_size=cfg.get("boundary_kernel_size", 3),
            boundary_weight=cfg.get("boundary_weight", 1.0),
            boundary_loss_mode=cfg.get("boundary_loss_mode", "weighted_bce"),
            use_tversky_loss=cfg.get("use_tversky_loss", False),
            tversky_alpha=cfg.get("tversky_alpha", 0.5),
            tversky_beta=cfg.get("tversky_beta", 0.5),
            tversky_weight=cfg.get("tversky_weight", 1.0),
            loss_mode=cfg.get("loss_mode", "bce"),
        )
        for item_idx, vis_item in enumerate(val_visual_examples):
            vis_item["epoch"] = epoch
            vis_item["dataset_index"] = val_indices[item_idx] if item_idx < len(val_indices) else vis_item["dataset_index"]

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_metrics['loss']:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"val_f1 {val_metrics['f1']:.4f} | "
            f"val_iou {val_metrics['iou']:.4f}"
        )
        print(
            "  loss terms | "
            f"train_chd {train_metrics['chd_loss']:.4f} | "
            f"train_tversky {train_metrics['tversky_loss']:.4f} | "
            f"train_aux {train_metrics['aux_rm_loss']:.4f} | "
            f"val_chd {val_metrics['chd_loss']:.4f} | "
            f"val_tversky {val_metrics['tversky_loss']:.4f} | "
            f"val_aux {val_metrics['aux_rm_loss']:.4f}"
        )
        print(
            "  boundary    | "
            f"train_mask_mean {train_metrics['boundary_mask_mean']:.4f} | "
            f"train_weight_mean {train_metrics['boundary_weight_mean']:.4f} | "
            f"val_mask_mean {val_metrics['boundary_mask_mean']:.4f} | "
            f"val_weight_mean {val_metrics['boundary_weight_mean']:.4f}"
        )
        print(
            "  train stats | "
            f"pred_pos_ratio {train_metrics['pred_pos_ratio']:.4f} | "
            f"gt_pos_ratio {train_metrics['gt_pos_ratio']:.4f} | "
            f"logit_mean {train_metrics['logit_mean']:.4f} | "
            f"logit_std {train_metrics['logit_std']:.4f} | "
            f"logit_min {train_metrics['logit_min']:.4f} | "
            f"logit_max {train_metrics['logit_max']:.4f} | "
            f"prob_mean {train_metrics['prob_mean']:.4f} | "
            f"prob_std {train_metrics['prob_std']:.4f} | "
            f"prob_min {train_metrics['prob_min']:.4f} | "
            f"prob_max {train_metrics['prob_max']:.4f} | "
            f"threshold {train_metrics['threshold']:.2f}"
        )
        print(
            "  val stats   | "
            f"pred_pos_ratio {val_metrics['pred_pos_ratio']:.4f} | "
            f"gt_pos_ratio {val_metrics['gt_pos_ratio']:.4f} | "
            f"logit_mean {val_metrics['logit_mean']:.4f} | "
            f"logit_std {val_metrics['logit_std']:.4f} | "
            f"logit_min {val_metrics['logit_min']:.4f} | "
            f"logit_max {val_metrics['logit_max']:.4f} | "
            f"prob_mean {val_metrics['prob_mean']:.4f} | "
            f"prob_std {val_metrics['prob_std']:.4f} | "
            f"prob_min {val_metrics['prob_min']:.4f} | "
            f"prob_max {val_metrics['prob_max']:.4f} | "
            f"threshold {val_metrics['threshold']:.2f}"
        )
        print(
            "  head stats  | "
            f"weight_norm {train_metrics['head_weight_norm']:.4f} | "
            f"bias_mean {train_metrics['head_bias_mean']:.4f} | "
            f"bias_value {train_metrics['head_bias_value']:.4f} | "
            f"grad_norm {train_metrics['head_grad_norm']:.6f} | "
            f"bias_grad_norm {train_metrics['head_bias_grad_norm']:.6f}"
        )
        if train_metrics.get("feature_stats"):
            for stage_name, stage_stats in train_metrics["feature_stats"].items():
                print("  train feat  | " + format_feature_stage_stats(stage_name, stage_stats))
            for stage_name, stage_stats in val_metrics.get("feature_stats", {}).items():
                print("  val feat    | " + format_feature_stage_stats(stage_name, stage_stats))
        if val_metrics.get("best_threshold") is not None:
            best_thr = val_metrics["best_threshold"]
            print(
                "  val sweep   | "
                f"best_threshold {best_thr['threshold']:.2f} | "
                f"f1 {best_thr['f1']:.4f} | "
                f"iou {best_thr['iou']:.4f} | "
                f"pred_pos_ratio {best_thr['pred_pos_ratio']:.4f}"
            )
            for item in val_metrics["threshold_sweep"]:
                print(
                    "    "
                    f"thr {item['threshold']:.2f} | "
                    f"f1 {item['f1']:.4f} | "
                    f"iou {item['iou']:.4f} | "
                    f"pred_pos_ratio {item['pred_pos_ratio']:.4f}"
                )

        current_fixed_is_best = val_metrics["f1"] > best_f1
        current_sweep_is_best = is_better_sweep_candidate(val_metrics.get("best_threshold"), best_sweep)
        current_sweep_best = val_metrics.get("best_threshold") if current_sweep_is_best else best_sweep
        current_sweep_best_epoch = epoch if current_sweep_is_best else best_sweep_epoch

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_chd_loss": train_metrics["chd_loss"],
            "train_aux_rm_loss": train_metrics["aux_rm_loss"],
            "train_tversky_loss": train_metrics["tversky_loss"],
            "train_boundary_mask_mean": train_metrics["boundary_mask_mean"],
            "train_boundary_weight_mean": train_metrics["boundary_weight_mean"],
            "val_loss": val_metrics["loss"],
            "val_chd_loss": val_metrics["chd_loss"],
            "val_aux_rm_loss": val_metrics["aux_rm_loss"],
            "val_tversky_loss": val_metrics["tversky_loss"],
            "val_boundary_mask_mean": val_metrics["boundary_mask_mean"],
            "val_boundary_weight_mean": val_metrics["boundary_weight_mean"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
            "train_pred_pos_ratio": train_metrics["pred_pos_ratio"],
            "train_gt_pos_ratio": train_metrics["gt_pos_ratio"],
            "train_logit_mean": train_metrics["logit_mean"],
            "train_logit_std": train_metrics["logit_std"],
            "train_logit_min": train_metrics["logit_min"],
            "train_logit_max": train_metrics["logit_max"],
            "train_prob_mean": train_metrics["prob_mean"],
            "train_prob_std": train_metrics["prob_std"],
            "train_prob_min": train_metrics["prob_min"],
            "train_prob_max": train_metrics["prob_max"],
            "val_pred_pos_ratio": val_metrics["pred_pos_ratio"],
            "val_gt_pos_ratio": val_metrics["gt_pos_ratio"],
            "val_logit_mean": val_metrics["logit_mean"],
            "val_logit_std": val_metrics["logit_std"],
            "val_logit_min": val_metrics["logit_min"],
            "val_logit_max": val_metrics["logit_max"],
            "val_prob_mean": val_metrics["prob_mean"],
            "val_prob_std": val_metrics["prob_std"],
            "val_prob_min": val_metrics["prob_min"],
            "val_prob_max": val_metrics["prob_max"],
            "head_weight_norm": train_metrics["head_weight_norm"],
            "head_bias_mean": train_metrics["head_bias_mean"],
            "head_bias_value": train_metrics["head_bias_value"],
            "head_grad_norm": train_metrics["head_grad_norm"],
            "head_bias_grad_norm": train_metrics["head_bias_grad_norm"],
            "pred_threshold": cfg["pred_threshold"],
            "use_pos_weight": cfg.get("use_pos_weight", True),
            "enable_aux_rm_head": cfg.get("enable_aux_rm_head", False),
            "aux_rm_loss_weight": cfg.get("aux_rm_loss_weight", 0.2),
            "aux_rm_loss_type": cfg.get("aux_rm_loss_type", "l1"),
            "aux_rm_detach_from_chd": cfg.get("aux_rm_detach_from_chd", False),
            "use_refine_head": cfg.get("use_refine_head", False),
            "refine_channels": cfg.get("refine_channels", 32),
            "refine_layers": cfg.get("refine_layers", 2),
            "refine_scale": cfg.get("refine_scale", 1.0),
            "use_boundary_loss": cfg.get("use_boundary_loss", False),
            "boundary_kernel_size": cfg.get("boundary_kernel_size", 3),
            "boundary_weight": cfg.get("boundary_weight", 1.0),
            "boundary_loss_mode": cfg.get("boundary_loss_mode", "weighted_bce"),
            "use_tversky_loss": cfg.get("use_tversky_loss", False),
            "loss_mode": cfg.get("loss_mode", "bce"),
            "tversky_alpha": cfg.get("tversky_alpha", 0.5),
            "tversky_beta": cfg.get("tversky_beta", 0.5),
            "tversky_weight": cfg.get("tversky_weight", 1.0),
            "val_best_threshold": None if val_metrics.get("best_threshold") is None else val_metrics["best_threshold"]["threshold"],
            "val_best_threshold_f1": None if val_metrics.get("best_threshold") is None else val_metrics["best_threshold"]["f1"],
            "val_best_threshold_iou": None if val_metrics.get("best_threshold") is None else val_metrics["best_threshold"]["iou"],
            "val_threshold_sweep": val_metrics.get("threshold_sweep", []),
            "best_sweep_f1": None if current_sweep_best is None else current_sweep_best["f1"],
            "best_sweep_threshold": None if current_sweep_best is None else current_sweep_best["threshold"],
            "best_sweep_iou": None if current_sweep_best is None else current_sweep_best["iou"],
            "best_sweep_epoch": current_sweep_best_epoch,
            "best_fixed_f1": max(best_f1, val_metrics["f1"]),
            "best_fixed_epoch": epoch if current_fixed_is_best else best_epoch,
        }
        history.append(epoch_metrics)
        append_jsonl(log_path, epoch_metrics)

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, epoch_metrics, cfg)
        if example is not None:
            save_prediction_artifact(output_dir / "last_pred.pt", example, epoch, epoch_metrics)
        save_split_examples("val", val_visual_examples, output_dir, cfg["pred_threshold"])
        if val_feature_example is not None:
            save_feature_visualizations("val", epoch, output_dir, val_feature_example)

        if current_fixed_is_best:
            best_f1 = val_metrics["f1"]
            best_metrics = epoch_metrics
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, epoch_metrics, cfg)
            if example is not None:
                save_prediction_artifact(output_dir / "best_pred.pt", example, epoch, epoch_metrics)

        if current_sweep_is_best:
            best_sweep = dict(val_metrics["best_threshold"])
            best_sweep_metrics = epoch_metrics
            best_sweep_epoch = epoch
            save_checkpoint(output_dir / "best_sweep.pt", model, optimizer, epoch, epoch_metrics, cfg)
            if example is not None:
                save_prediction_artifact(output_dir / "best_sweep_pred.pt", example, epoch, epoch_metrics)

        fixed_best_msg = (
            f"epoch {best_epoch:03d} | f1 {best_f1:.4f}"
            if best_metrics is not None and best_epoch is not None
            else "not set"
        )
        sweep_best_msg = (
            f"epoch {best_sweep_epoch:03d} | thr {best_sweep['threshold']:.2f} | "
            f"f1 {best_sweep['f1']:.4f} | iou {best_sweep['iou']:.4f}"
            if best_sweep is not None and best_sweep_epoch is not None
            else "not set"
        )
        print(f"  best fixed | {fixed_best_msg}")
        print(f"  best sweep | {sweep_best_msg}")

    with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    save_training_curves(output_dir, history)

    if best_metrics is not None:
        print(
            "best | "
            f"val_f1 {best_metrics['val_f1']:.4f} | "
            f"val_iou {best_metrics['val_iou']:.4f} | "
            f"val_loss {best_metrics['val_loss']:.4f}"
        )
        with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)
    if best_sweep_metrics is not None and best_sweep is not None and best_sweep_epoch is not None:
        print(
            "best_sweep | "
            f"epoch {best_sweep_epoch:03d} | "
            f"threshold {best_sweep['threshold']:.2f} | "
            f"f1 {best_sweep['f1']:.4f} | "
            f"iou {best_sweep['iou']:.4f}"
        )
        with open(output_dir / "best_sweep_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_sweep_metrics, f, indent=2)

    if test_loader is not None and (output_dir / "best.pt").exists():
        best_ckpt = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        test_metrics, test_example, test_visual_examples, test_feature_example = validate(
            model,
            test_loader,
            pos_weight,
            device,
            task_id_value,
            cfg["gt_type"],
            cfg["pred_threshold"],
            atrc_scorer=atrc_scorer,
            max_visual_examples=cfg.get("num_visual_examples", 4),
            threshold_sweep_values=threshold_sweep_values,
            enable_aux_rm_head=cfg.get("enable_aux_rm_head", False),
            aux_rm_loss_weight=cfg.get("aux_rm_loss_weight", 0.2),
            aux_rm_loss_type=cfg.get("aux_rm_loss_type", "l1"),
            aux_rm_require_target=cfg.get("aux_rm_require_target", True),
            use_boundary_loss=cfg.get("use_boundary_loss", False),
            boundary_kernel_size=cfg.get("boundary_kernel_size", 3),
            boundary_weight=cfg.get("boundary_weight", 1.0),
            boundary_loss_mode=cfg.get("boundary_loss_mode", "weighted_bce"),
            use_tversky_loss=cfg.get("use_tversky_loss", False),
            tversky_alpha=cfg.get("tversky_alpha", 0.5),
            tversky_beta=cfg.get("tversky_beta", 0.5),
            tversky_weight=cfg.get("tversky_weight", 1.0),
            loss_mode=cfg.get("loss_mode", "bce"),
        )
        for item_idx, vis_item in enumerate(test_visual_examples):
            vis_item["epoch"] = best_ckpt["epoch"]
            vis_item["dataset_index"] = test_indices[item_idx] if item_idx < len(test_indices) else vis_item["dataset_index"]
        print(
            "test | "
            f"loss {test_metrics['loss']:.4f} | "
            f"f1 {test_metrics['f1']:.4f} | "
            f"iou {test_metrics['iou']:.4f}"
        )
        print(
            "  test loss  | "
            f"chd {test_metrics['chd_loss']:.4f} | "
            f"aux {test_metrics['aux_rm_loss']:.4f}"
        )
        print(
            "  test fixed | "
            f"threshold {cfg['pred_threshold']:.2f} | "
            f"f1 {test_metrics['f1']:.4f} | "
            f"iou {test_metrics['iou']:.4f}"
        )
        print(
            "  test stats  | "
            f"pred_pos_ratio {test_metrics['pred_pos_ratio']:.4f} | "
            f"gt_pos_ratio {test_metrics['gt_pos_ratio']:.4f} | "
            f"prob_mean {test_metrics['prob_mean']:.4f} | "
            f"prob_std {test_metrics['prob_std']:.4f} | "
            f"prob_min {test_metrics['prob_min']:.4f} | "
            f"prob_max {test_metrics['prob_max']:.4f} | "
            f"threshold {test_metrics['threshold']:.2f}"
        )
        if test_metrics.get("best_threshold") is not None:
            best_thr = test_metrics["best_threshold"]
            print(
                "  test sweep  | "
                f"best_threshold {best_thr['threshold']:.2f} | "
                f"f1 {best_thr['f1']:.4f} | "
                f"iou {best_thr['iou']:.4f} | "
                f"pred_pos_ratio {best_thr['pred_pos_ratio']:.4f}"
            )
        if test_example is not None:
            save_prediction_artifact(output_dir / "test_pred.pt", test_example, best_ckpt["epoch"], test_metrics)
        save_split_examples("test", test_visual_examples, output_dir, cfg["pred_threshold"])
        if test_feature_example is not None:
            save_feature_visualizations("test", best_ckpt["epoch"], output_dir, test_feature_example)

        sweep_test_metrics = None
        sweep_test_example = None
        sweep_test_threshold = None
        best_sweep_ckpt_path = output_dir / "best_sweep.pt"
        best_sweep_metrics_path = output_dir / "best_sweep_metrics.json"
        if best_sweep_ckpt_path.exists() and best_sweep_metrics_path.exists():
            with open(best_sweep_metrics_path, "r", encoding="utf-8") as f:
                best_sweep_meta = json.load(f)
            sweep_test_threshold = best_sweep_meta.get("best_sweep_threshold")
            if sweep_test_threshold is None:
                raise KeyError(f"best_sweep_threshold was missing from {best_sweep_metrics_path}")

            best_sweep_ckpt = torch.load(best_sweep_ckpt_path, map_location=device)
            model.load_state_dict(best_sweep_ckpt["model_state"])
            sweep_test_metrics, sweep_test_example, sweep_test_visual_examples, sweep_test_feature_example = validate(
                model,
                test_loader,
                pos_weight,
                device,
                task_id_value,
                cfg["gt_type"],
                float(sweep_test_threshold),
                atrc_scorer=atrc_scorer,
                max_visual_examples=cfg.get("num_visual_examples", 4),
                threshold_sweep_values=[],
                enable_aux_rm_head=cfg.get("enable_aux_rm_head", False),
                aux_rm_loss_weight=cfg.get("aux_rm_loss_weight", 0.2),
                aux_rm_loss_type=cfg.get("aux_rm_loss_type", "l1"),
                aux_rm_require_target=cfg.get("aux_rm_require_target", True),
            use_boundary_loss=cfg.get("use_boundary_loss", False),
            boundary_kernel_size=cfg.get("boundary_kernel_size", 3),
            boundary_weight=cfg.get("boundary_weight", 1.0),
            boundary_loss_mode=cfg.get("boundary_loss_mode", "weighted_bce"),
            use_tversky_loss=cfg.get("use_tversky_loss", False),
            tversky_alpha=cfg.get("tversky_alpha", 0.5),
            tversky_beta=cfg.get("tversky_beta", 0.5),
            tversky_weight=cfg.get("tversky_weight", 1.0),
            loss_mode=cfg.get("loss_mode", "bce"),
        )
            for item_idx, vis_item in enumerate(sweep_test_visual_examples):
                vis_item["epoch"] = best_sweep_ckpt["epoch"]
                vis_item["dataset_index"] = test_indices[item_idx] if item_idx < len(test_indices) else vis_item["dataset_index"]

            print(
                "test_sweep | "
                f"loss {sweep_test_metrics['loss']:.4f} | "
                f"f1 {sweep_test_metrics['f1']:.4f} | "
                f"iou {sweep_test_metrics['iou']:.4f}"
            )
            print(
                "  test sweep | "
                f"threshold {float(sweep_test_threshold):.2f} | "
                f"f1 {sweep_test_metrics['f1']:.4f} | "
                f"iou {sweep_test_metrics['iou']:.4f}"
            )
            print(
                "  sweep stats | "
                f"pred_pos_ratio {sweep_test_metrics['pred_pos_ratio']:.4f} | "
                f"gt_pos_ratio {sweep_test_metrics['gt_pos_ratio']:.4f} | "
                f"prob_mean {sweep_test_metrics['prob_mean']:.4f} | "
                f"prob_std {sweep_test_metrics['prob_std']:.4f} | "
                f"prob_min {sweep_test_metrics['prob_min']:.4f} | "
                f"prob_max {sweep_test_metrics['prob_max']:.4f}"
            )

            sweep_test_metrics_out = dict(sweep_test_metrics)
            sweep_test_metrics_out["test_sweep_f1"] = sweep_test_metrics["f1"]
            sweep_test_metrics_out["test_sweep_iou"] = sweep_test_metrics["iou"]
            sweep_test_metrics_out["test_sweep_threshold"] = float(sweep_test_threshold)
            sweep_test_metrics_out["source_checkpoint"] = str(best_sweep_ckpt_path)
            sweep_test_metrics_out["source_epoch"] = int(best_sweep_ckpt["epoch"])
            with open(output_dir / "test_sweep_metrics.json", "w", encoding="utf-8") as f:
                json.dump(sweep_test_metrics_out, f, indent=2)
            if sweep_test_example is not None:
                save_prediction_artifact(output_dir / "test_sweep_pred.pt", sweep_test_example, best_sweep_ckpt["epoch"], sweep_test_metrics_out)
            save_split_examples("test_sweep", sweep_test_visual_examples, output_dir, float(sweep_test_threshold))
            if sweep_test_feature_example is not None:
                save_feature_visualizations("test_sweep", best_sweep_ckpt["epoch"], output_dir, sweep_test_feature_example)

        test_metrics_out = dict(test_metrics)
        test_metrics_out["test_fixed_f1"] = test_metrics["f1"]
        test_metrics_out["test_fixed_iou"] = test_metrics["iou"]
        test_metrics_out["test_fixed_threshold"] = float(cfg["pred_threshold"])
        test_metrics_out["test_sweep_f1"] = None if sweep_test_metrics is None else sweep_test_metrics["f1"]
        test_metrics_out["test_sweep_iou"] = None if sweep_test_metrics is None else sweep_test_metrics["iou"]
        test_metrics_out["test_sweep_threshold"] = None if sweep_test_threshold is None else float(sweep_test_threshold)
        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics_out, f, indent=2)

        test_summary = {
            "record_type": "test_summary",
            "epoch": int(best_ckpt["epoch"]),
            "test_fixed_f1": test_metrics["f1"],
            "test_fixed_iou": test_metrics["iou"],
            "test_fixed_threshold": float(cfg["pred_threshold"]),
            "test_sweep_f1": None if sweep_test_metrics is None else sweep_test_metrics["f1"],
            "test_sweep_iou": None if sweep_test_metrics is None else sweep_test_metrics["iou"],
            "test_sweep_threshold": None if sweep_test_threshold is None else float(sweep_test_threshold),
        }
        history.append(test_summary)
        append_jsonl(log_path, test_summary)
        with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

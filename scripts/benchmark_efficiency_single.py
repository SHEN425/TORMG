import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.dataset_cache import CacheDataset
from scripts.train_mvp import build_model, load_config, resolve_device


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark one model for latency/memory/FLOPs/params.")
    p.add_argument("--config", type=str, required=True, help="Path to model config.")
    p.add_argument("--model_name", type=str, default="", help="Override model_name in output row.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--checkpoint_name", type=str, default="best.pt")
    p.add_argument("--fallback_checkpoint_name", type=str, default="last.pt")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--csv_output", type=str, required=True)
    return p.parse_args()


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_checkpoint(output_dir: Path, primary: str, fallback: str) -> Path:
    first = output_dir / primary
    if first.exists():
        return first
    second = output_dir / fallback
    if second.exists():
        return second
    raise FileNotFoundError(f"No checkpoint found in {output_dir} ({primary}/{fallback})")


def build_unified_inputs(cache_dir: str, batch_size: int, device: torch.device):
    dataset = CacheDataset(cache_dir=cache_dir, gt_type="cache", physical_gt_dir=None, fallback_to_cache_gt=True)
    sample = dataset[0]
    return {
        "city": sample["city"].unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        "meas_xy": sample["meas_xy"].unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        "meas_v": sample["meas_v"].unsqueeze(0).repeat(batch_size, 1).to(device),
        "bs_xy": sample["bs_xy"].unsqueeze(0).repeat(batch_size, 1, 1).to(device),
    }


def run_forward(model: torch.nn.Module, inputs):
    return model(inputs["meas_xy"], inputs["meas_v"], inputs["bs_xy"], city=inputs["city"])


def measure_flops(model: torch.nn.Module, inputs, device: torch.device) -> int:
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    maybe_sync(device)
    with torch.no_grad():
        with profile(activities=acts, with_flops=True, profile_memory=False, record_shapes=False) as prof:
            _ = run_forward(model, inputs)
    maybe_sync(device)
    total = 0
    for ev in prof.key_averages():
        if getattr(ev, "flops", 0):
            total += int(ev.flops)
    return int(total)


def measure_latency_memory(model: torch.nn.Module, inputs, device: torch.device, warmup: int, iters: int):
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = run_forward(model, inputs)
        maybe_sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = run_forward(model, inputs)
        maybe_sync(device)
        elapsed = time.perf_counter() - start
    latency_ms = (elapsed / max(1, iters)) * 1000.0
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    return latency_ms, peak_mb


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    model = build_model(cfg).to(device)
    ckpt_path = resolve_checkpoint(Path(cfg["output_dir"]), args.checkpoint_name, args.fallback_checkpoint_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    inputs = build_unified_inputs(cfg["cache_dir"], args.batch_size, device)
    params = count_params(model)
    flops = measure_flops(model, inputs, device)
    latency_ms, peak_mb = measure_latency_memory(model, inputs, device, args.warmup, args.iters)

    row = {
        "model_name": args.model_name if args.model_name else cfg.get("model_name", "unknown"),
        "latency_ms": float(latency_ms),
        "peak_gpu_memory_mb": float(peak_mb),
        "flops": int(flops),
        "params": int(params),
        "checkpoint": str(ckpt_path),
        "config_path": str(cfg_path),
        "resolved_output_dir": cfg["output_dir"],
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
    }

    out_json = Path(args.output).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": [row]}, f, indent=2)

    out_csv = Path(args.csv_output).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

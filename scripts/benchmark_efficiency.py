import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.dataset_cache import CacheDataset
from scripts.train_mvp import build_model, load_config, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark latency/memory/FLOPs/params for formal models.")
    parser.add_argument(
        "--allgrid_config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_mvp_soft_allgrid_server_full.json"),
        help="Path to allgrid formal config.",
    )
    parser.add_argument(
        "--envgrid_plus_config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_mvp_soft_envgrid_plus_server_full.json"),
        help="Path to envgrid_plus formal config.",
    )
    parser.add_argument(
        "--unet_config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_unet_soft_server_full.json"),
        help="Path to direct-CHD UNet formal config.",
    )
    parser.add_argument(
        "--rm_unet_config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_rm_unet_soft_server_full.json"),
        help="Path to RM-reconstruction UNet formal config.",
    )
    parser.add_argument(
        "--fcn_config",
        type=str,
        default="",
        help="Optional path to FCN direct-CHD config. When provided, FCN efficiency will be benchmarked as an extra row.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Benchmark device, e.g. cuda/cpu/auto.")
    parser.add_argument("--batch_size", type=int, default=1, help="Unified batch size for all models.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best.pt",
        help="Preferred checkpoint file name under each config output_dir.",
    )
    parser.add_argument(
        "--fallback_checkpoint_name",
        type=str,
        default="last.pt",
        help="Fallback checkpoint file name under each config output_dir.",
    )
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        help="If no checkpoint is found, benchmark random-initialized model instead of failing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "formal" / "efficiency_benchmark" / "efficiency_results_four_models.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "formal" / "efficiency_benchmark" / "efficiency_results_four_models.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_checkpoint(output_dir: Path, primary: str, fallback: str) -> Optional[Path]:
    first = output_dir / primary
    if first.exists():
        return first
    second = output_dir / fallback
    if second.exists():
        return second
    return None


def load_model_from_config(
    config_path: Path,
    device: torch.device,
    checkpoint_name: str,
    fallback_checkpoint_name: str,
    allow_random_init: bool,
) -> Tuple[torch.nn.Module, dict, Optional[Path]]:
    cfg = load_config(config_path)
    model = build_model(cfg).to(device)

    ckpt_path = resolve_checkpoint(Path(cfg["output_dir"]), checkpoint_name, fallback_checkpoint_name)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif not allow_random_init:
        raise FileNotFoundError(
            f"No checkpoint found for config={config_path}. "
            f"Tried: {Path(cfg['output_dir']) / checkpoint_name} and {Path(cfg['output_dir']) / fallback_checkpoint_name}."
        )

    model.eval()
    return model, cfg, ckpt_path


def build_unified_inputs(cache_dir: str, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    dataset = CacheDataset(cache_dir=cache_dir, gt_type="cache", physical_gt_dir=None, fallback_to_cache_gt=True)
    sample = dataset[0]

    city = sample["city"].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    meas_xy = sample["meas_xy"].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    meas_v = sample["meas_v"].unsqueeze(0).repeat(batch_size, 1).to(device)
    bs_xy = sample["bs_xy"].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    return {"city": city, "meas_xy": meas_xy, "meas_v": meas_v, "bs_xy": bs_xy}


def run_forward(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]):
    return model(
        inputs["meas_xy"],
        inputs["meas_v"],
        inputs["bs_xy"],
        city=inputs["city"],
    )


def measure_flops(model: torch.nn.Module, inputs: Dict[str, torch.Tensor], device: torch.device) -> int:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    maybe_sync(device)
    with torch.no_grad():
        with profile(activities=activities, with_flops=True, profile_memory=False, record_shapes=False) as prof:
            _ = run_forward(model, inputs)
    maybe_sync(device)

    total_flops = 0
    for event in prof.key_averages():
        if getattr(event, "flops", 0):
            total_flops += int(event.flops)
    return int(total_flops)


def measure_latency_and_peak_memory(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
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


def format_int(n: int) -> str:
    return f"{n:,}"


def print_markdown_table(rows):
    headers = ["model_name", "latency_ms", "peak_gpu_memory_mb", "flops", "params", "checkpoint"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    f"{row['latency_ms']:.3f}",
                    f"{row['peak_gpu_memory_mb']:.2f}",
                    format_int(int(row["flops"])),
                    format_int(int(row["params"])),
                    row["checkpoint"] if row["checkpoint"] is not None else "random_init",
                ]
            )
            + " |"
        )


def save_csv(rows, csv_path: Path):
    fieldnames = [
        "model_name",
        "latency_ms",
        "peak_gpu_memory_mb",
        "flops",
        "params",
        "checkpoint",
        "config_path",
        "resolved_output_dir",
        "device",
        "batch_size",
        "warmup",
        "iters",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def benchmark_one(
    name: str,
    config_path: Path,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int,
    iters: int,
    checkpoint_name: str,
    fallback_checkpoint_name: str,
    allow_random_init: bool,
):
    model, cfg, ckpt_path = load_model_from_config(
        config_path=config_path,
        device=device,
        checkpoint_name=checkpoint_name,
        fallback_checkpoint_name=fallback_checkpoint_name,
        allow_random_init=allow_random_init,
    )

    params = count_params(model)
    flops = measure_flops(model, inputs, device)
    latency_ms, peak_mb = measure_latency_and_peak_memory(model, inputs, device, warmup=warmup, iters=iters)

    return {
        "model_name": name,
        "config_path": str(config_path),
        "resolved_output_dir": cfg["output_dir"],
        "checkpoint": None if ckpt_path is None else str(ckpt_path),
        "device": str(device),
        "batch_size": int(inputs["city"].shape[0]),
        "latency_ms": float(latency_ms),
        "peak_gpu_memory_mb": float(peak_mb),
        "flops": int(flops),
        "params": int(params),
        "warmup": int(warmup),
        "iters": int(iters),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)

    allgrid_config = Path(args.allgrid_config).resolve()
    envgrid_plus_config = Path(args.envgrid_plus_config).resolve()
    unet_config = Path(args.unet_config).resolve()
    rm_unet_config = Path(args.rm_unet_config).resolve()
    fcn_config = Path(args.fcn_config).resolve() if args.fcn_config else None

    # Use one unified canonical input source so all models run under identical input tensors.
    allgrid_cfg = load_config(allgrid_config)
    inputs = build_unified_inputs(
        cache_dir=allgrid_cfg["cache_dir"],
        batch_size=args.batch_size,
        device=device,
    )

    rows = []
    rows.append(
        benchmark_one(
            name="mvp_crossattn_envgrid_plus",
            config_path=envgrid_plus_config,
            inputs=inputs,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            checkpoint_name=args.checkpoint_name,
            fallback_checkpoint_name=args.fallback_checkpoint_name,
            allow_random_init=args.allow_random_init,
        )
    )
    rows.append(
        benchmark_one(
            name="mvp_crossattn_allgrid",
            config_path=allgrid_config,
            inputs=inputs,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            checkpoint_name=args.checkpoint_name,
            fallback_checkpoint_name=args.fallback_checkpoint_name,
            allow_random_init=args.allow_random_init,
        )
    )
    rows.append(
        benchmark_one(
            name="unet_baseline",
            config_path=unet_config,
            inputs=inputs,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            checkpoint_name=args.checkpoint_name,
            fallback_checkpoint_name=args.fallback_checkpoint_name,
            allow_random_init=args.allow_random_init,
        )
    )
    rows.append(
        benchmark_one(
            name="rm_unet_baseline",
            config_path=rm_unet_config,
            inputs=inputs,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            checkpoint_name=args.checkpoint_name,
            fallback_checkpoint_name=args.fallback_checkpoint_name,
            allow_random_init=args.allow_random_init,
        )
    )
    if fcn_config is not None:
        rows.append(
            benchmark_one(
                name="fcn_baseline",
                config_path=fcn_config,
                inputs=inputs,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                checkpoint_name=args.checkpoint_name,
                fallback_checkpoint_name=args.fallback_checkpoint_name,
                allow_random_init=args.allow_random_init,
            )
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "formal_efficiency_four_models",
        "timestamp_unix": time.time(),
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = Path(args.csv_output).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(rows, csv_path)

    print(f"saved: {output_path}")
    print(f"saved: {csv_path}")
    print_markdown_table(rows)


if __name__ == "__main__":
    main()

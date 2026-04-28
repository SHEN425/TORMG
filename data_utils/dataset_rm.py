import ntpath
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class RMReconstructionDataset(Dataset):
    """
    Read cached MVP inputs and supervise with full-resolution radio maps from
    physical GT artifacts.
    """

    def __init__(
        self,
        cache_dir: str,
        physical_gt_dir: str,
        fallback_to_cache_gt: bool = False,
        normalize_rm: bool = True,
        meas_keep_count: Optional[int] = None,
        meas_sample_mode: str = "random",
        meas_sample_seed: int = 0,
        meas_sample_deterministic: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("sample_*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {self.cache_dir}")

        self.physical_gt_dir = Path(physical_gt_dir)
        if not self.physical_gt_dir.exists():
            raise FileNotFoundError(f"physical_gt_dir not found: {self.physical_gt_dir}")

        self.fallback_to_cache_gt = bool(fallback_to_cache_gt)
        self.normalize_rm = bool(normalize_rm)

        self.meas_keep_count = None if meas_keep_count is None else int(meas_keep_count)
        if self.meas_keep_count is not None and self.meas_keep_count <= 0:
            self.meas_keep_count = None
        self.meas_sample_mode = str(meas_sample_mode).lower()
        if self.meas_sample_mode not in {"random", "first"}:
            raise ValueError(f"Unsupported meas_sample_mode={meas_sample_mode}. Use 'random' or 'first'.")
        self.meas_sample_seed = int(meas_sample_seed)
        self.meas_sample_deterministic = bool(meas_sample_deterministic)

        self.physical_gt_lookup = self._build_physical_gt_lookup()

    def __len__(self):
        return len(self.files)

    def _build_physical_gt_lookup(self):
        lookup = {}
        for path in sorted(self.physical_gt_dir.rglob("*_physical_gt.pt")):
            d = torch.load(path, map_location="cpu")
            meta = d.get("meta", {})
            map_name = metadata_path_name(meta.get("map_path"))
            gain_name = metadata_path_name(meta.get("gain_path"))
            if map_name and gain_name:
                lookup[(map_name, gain_name)] = path
        return lookup

    def _select_radio_target(self, cache_item: dict):
        meta = cache_item.get("meta", {})
        key = (meta.get("map"), meta.get("gain"))
        gt_path = self.physical_gt_lookup.get(key)

        if gt_path is None:
            if self.fallback_to_cache_gt:
                # Fallback is only for debugging; this is NOT a real RM target.
                y_proxy = cache_item["y_grid"].float().unsqueeze(0).unsqueeze(0)
                y_proxy = torch.nn.functional.interpolate(y_proxy, size=(256, 256), mode="nearest").squeeze(0).squeeze(0)
                return y_proxy
            raise KeyError(f"No physical GT artifact matched cache sample with key={key}")

        gt_artifact = torch.load(gt_path, map_location="cpu")
        radio = gt_artifact["radio"].float()
        if self.normalize_rm:
            radio = (radio / 255.0).clamp(0.0, 1.0)
        return radio

    def _select_meas_indices(self, n: int, idx: int):
        if self.meas_keep_count is None or self.meas_keep_count >= n:
            return None
        k = max(1, int(self.meas_keep_count))

        if self.meas_sample_mode == "first":
            return torch.arange(k, dtype=torch.long)

        if self.meas_sample_deterministic:
            g = torch.Generator()
            g.manual_seed(self.meas_sample_seed + idx * 1000003)
            perm = torch.randperm(n, generator=g)
        else:
            perm = torch.randperm(n)

        sel = perm[:k]
        sel, _ = torch.sort(sel)
        return sel

    def __getitem__(self, idx: int):
        d = torch.load(self.files[idx], map_location="cpu")

        city = d["city_mask"].float()
        meas_xy = d["meas_xy"].long()
        meas_v = d["meas_v"].float() / 255.0
        bs_xy = d["bs_xy"].float()
        y_rm = self._select_radio_target(d)

        keep_idx = self._select_meas_indices(meas_xy.shape[0], idx)
        if keep_idx is not None:
            meas_xy = meas_xy[keep_idx]
            meas_v = meas_v[keep_idx]

        h, w = city.shape

        meas_xy_n = meas_xy.float()
        meas_xy_n[:, 0] = meas_xy_n[:, 0] / (w - 1)
        meas_xy_n[:, 1] = meas_xy_n[:, 1] / (h - 1)

        bs_xy_n = bs_xy.clone()
        bs_xy_n[:, 0] = bs_xy_n[:, 0] / (w - 1)
        bs_xy_n[:, 1] = bs_xy_n[:, 1] / (h - 1)

        return {
            "city": city,
            "meas_xy": meas_xy_n,
            "meas_v": meas_v,
            "bs_xy": bs_xy_n,
            "y_rm": y_rm,
        }


def metadata_path_name(path_value) -> str:
    if path_value is None:
        return ""
    return ntpath.basename(str(path_value))

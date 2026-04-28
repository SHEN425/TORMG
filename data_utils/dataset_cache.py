import ntpath
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """
    Read cached MVP samples and expose selectable supervision targets.

    By default this dataset returns only CHD target (y_grid). When
    return_rm_target=True, it additionally returns an RM target (y_rm)
    derived from physical GT artifacts.
    """

    def __init__(
        self,
        cache_dir: str,
        gt_type: str = "soft",
        physical_gt_dir: Optional[str] = None,
        fallback_to_cache_gt: bool = True,
        return_rm_target: bool = False,
        normalize_rm_target: bool = True,
        meas_keep_count: Optional[int] = None,
        meas_sample_mode: str = "random",
        meas_sample_seed: int = 0,
        meas_sample_deterministic: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("sample_*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {self.cache_dir}")

        self.gt_type = gt_type
        self.physical_gt_dir = None if physical_gt_dir is None else Path(physical_gt_dir)
        self.fallback_to_cache_gt = fallback_to_cache_gt
        self.return_rm_target = bool(return_rm_target)
        self.normalize_rm_target = bool(normalize_rm_target)
        self.allowed_gt_types = {"cache", "hard", "soft", "refined"}
        if self.gt_type not in self.allowed_gt_types:
            raise ValueError(f"Unsupported gt_type: {self.gt_type}. Expected one of {sorted(self.allowed_gt_types)}")

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
        if self.physical_gt_dir is None or not self.physical_gt_dir.exists():
            return lookup

        for path in sorted(self.physical_gt_dir.rglob("*_physical_gt.pt")):
            d = torch.load(path, map_location="cpu")
            meta = d.get("meta", {})
            map_name = metadata_path_name(meta.get("map_path"))
            gain_name = metadata_path_name(meta.get("gain_path"))
            if map_name and gain_name:
                lookup[(map_name, gain_name)] = path
        return lookup

    def _downsample_gt(self, gt_full: torch.Tensor, out_hw):
        gt = gt_full.float().unsqueeze(0).unsqueeze(0)
        gt = F.interpolate(gt, size=out_hw, mode="area").squeeze(0).squeeze(0)
        if self.gt_type in {"hard", "refined"}:
            return (gt > 0.5).float()
        return gt.clamp(0.0, 1.0)

    def _downsample_rm(self, radio_full: torch.Tensor, out_hw):
        radio = radio_full.float().unsqueeze(0).unsqueeze(0)
        radio = F.interpolate(radio, size=out_hw, mode="area").squeeze(0).squeeze(0)
        return radio.clamp(0.0, 1.0)

    def _resolve_physical_gt_path(self, cache_item: dict):
        meta = cache_item.get("meta", {})
        key = (meta.get("map"), meta.get("gain"))
        return self.physical_gt_lookup.get(key)

    def _select_target(self, cache_item: dict):
        y_cache = cache_item["y_grid"].float()
        if self.gt_type == "cache":
            return y_cache

        gt_path = self._resolve_physical_gt_path(cache_item)
        if gt_path is None:
            if self.fallback_to_cache_gt:
                return y_cache
            raise KeyError(f"No physical GT artifact matched cache sample with key={cache_item.get('meta', {})}")

        gt_artifact = torch.load(gt_path, map_location="cpu")
        if self.gt_type == "hard":
            gt_full = gt_artifact["hard_gt"]
        elif self.gt_type == "soft":
            gt_full = gt_artifact["soft_gt"]
        else:
            gt_full = gt_artifact["refined_gt"]
        return self._downsample_gt(gt_full, y_cache.shape)

    def _select_rm_target(self, cache_item: dict, out_hw):
        gt_path = self._resolve_physical_gt_path(cache_item)
        if gt_path is None:
            raise KeyError(
                "No physical GT artifact matched cache sample while return_rm_target=True. "
                f"sample meta={cache_item.get('meta', {})}"
            )

        gt_artifact = torch.load(gt_path, map_location="cpu")
        if "radio" not in gt_artifact:
            raise KeyError(f"Expected key 'radio' in physical GT artifact: {gt_path}")

        radio = gt_artifact["radio"].float()
        if self.normalize_rm_target:
            radio = (radio / 255.0).clamp(0.0, 1.0)
        return self._downsample_rm(radio, out_hw)

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
        y = self._select_target(d)

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

        out = {
            "city": city,
            "meas_xy": meas_xy_n,
            "meas_v": meas_v,
            "bs_xy": bs_xy_n,
            "y_grid": y,
        }
        if self.return_rm_target:
            out["y_rm"] = self._select_rm_target(d, out_hw=y.shape)
        return out


def metadata_path_name(path_value) -> str:
    """
    Extract the filename from metadata path strings that may use either
    Linux-style or Windows-style separators.
    """
    if path_value is None:
        return ""
    return ntpath.basename(str(path_value))

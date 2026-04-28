import torch
import torch.nn as nn
import torch.nn.functional as F


class ATRCScorer(nn.Module):
    """
    Minimal ATRC v1 scorer:
    - measurement density
    - BS density
    - urban complexity from city mask
    """

    def __init__(
        self,
        region_grid_size=8,
        num_levels=3,
        measurement_weight=1.0,
        bs_weight=1.0,
        urban_weight=1.0,
    ):
        super().__init__()
        self.region_grid_size = region_grid_size
        self.num_levels = num_levels

        weight = torch.tensor(
            [measurement_weight, bs_weight, urban_weight],
            dtype=torch.float32,
        )
        weight = weight / weight.sum().clamp_min(1e-8)
        self.register_buffer("metric_weight", weight, persistent=False)

        ys = (torch.arange(region_grid_size, dtype=torch.float32) + 0.5) / region_grid_size
        xs = (torch.arange(region_grid_size, dtype=torch.float32) + 0.5) / region_grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        region_xy = torch.stack([gx, gy], dim=-1)
        self.register_buffer("region_xy", region_xy, persistent=False)

    def normalize_map(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        x_min = x_flat.min(dim=1, keepdim=True).values
        x_max = x_flat.max(dim=1, keepdim=True).values
        x_norm = (x_flat - x_min) / (x_max - x_min).clamp_min(1e-8)
        return x_norm.view_as(x)

    def points_to_region_density(self, points):
        """
        points: (B,N,2) normalized to [0,1]
        return: (B,R,R)
        """
        b, n, _ = points.shape
        r = self.region_grid_size

        x = torch.floor(points[..., 0] * r).long().clamp(0, r - 1)
        y = torch.floor(points[..., 1] * r).long().clamp(0, r - 1)
        region_idx = y * r + x

        density = torch.zeros(b, r * r, device=points.device, dtype=points.dtype)
        ones = torch.ones(b, n, device=points.device, dtype=points.dtype)
        density.scatter_add_(1, region_idx, ones)
        return density.view(b, r, r)

    def compute_measurement_density(self, meas_xy):
        return self.points_to_region_density(meas_xy)

    def compute_bs_density(self, bs_xy):
        return self.points_to_region_density(bs_xy)

    def compute_urban_complexity(self, city):
        """
        city: (B,H,W), building/city mask
        return: (B,R,R)
        """
        city = city.float()
        dx = torch.abs(city[:, :, 1:] - city[:, :, :-1])
        dy = torch.abs(city[:, 1:, :] - city[:, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = 0.5 * (dx + dy)
        return F.adaptive_avg_pool2d(edge.unsqueeze(1), (self.region_grid_size, self.region_grid_size)).squeeze(1)

    def combine_scores(self, measurement_density, bs_density, urban_complexity):
        meas_n = self.normalize_map(measurement_density)
        bs_n = self.normalize_map(bs_density)
        urban_n = self.normalize_map(urban_complexity)

        importance = (
            self.metric_weight[0] * meas_n
            + self.metric_weight[1] * bs_n
            + self.metric_weight[2] * urban_n
        )
        importance = self.normalize_map(importance)
        return {
            "measurement_density": meas_n,
            "bs_density": bs_n,
            "urban_complexity": urban_n,
            "importance": importance,
        }

    def importance_to_level(self, importance):
        if self.num_levels <= 1:
            return torch.zeros_like(importance, dtype=torch.long)
        levels = torch.floor(importance * self.num_levels).long()
        return levels.clamp(max=self.num_levels - 1)

    def forward(self, city, meas_xy, bs_xy):
        measurement_density = self.compute_measurement_density(meas_xy)
        bs_density = self.compute_bs_density(bs_xy)
        urban_complexity = self.compute_urban_complexity(city)

        out = self.combine_scores(measurement_density, bs_density, urban_complexity)
        out["refinement_level"] = self.importance_to_level(out["importance"])
        out["region_xy"] = self.region_xy
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class SanitySoftBaseline(nn.Module):
    """
    A tiny 2D baseline for sanity-checking whether the single-BS + soft-GT task
    is learnable at all under the current data/supervision pipeline.

    Inputs are rasterized to a coarse 2D grid:
    - city mask
    - measurement count
    - measurement mean value
    - BS one-hot location
    - BS distance map

    The resulting channels are passed through a shallow CNN to predict the soft
    target at the same grid resolution used by training.
    """

    def __init__(self, grid_size=64, hidden_channels=32):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", gx, persistent=False)
        self.register_buffer("grid_y", gy, persistent=False)

        in_channels = 5
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.coverage_hole_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def rasterize_measurements(self, meas_xy, meas_v):
        B, N, _ = meas_xy.shape
        G = self.grid_size
        x = torch.round(meas_xy[..., 0] * (G - 1)).long().clamp(0, G - 1)
        y = torch.round(meas_xy[..., 1] * (G - 1)).long().clamp(0, G - 1)
        flat_idx = y * G + x

        count_map = torch.zeros(B, G * G, device=meas_xy.device, dtype=meas_v.dtype)
        sum_map = torch.zeros(B, G * G, device=meas_xy.device, dtype=meas_v.dtype)
        ones = torch.ones_like(meas_v)
        count_map.scatter_add_(1, flat_idx, ones)
        sum_map.scatter_add_(1, flat_idx, meas_v)

        count_map = count_map.view(B, 1, G, G)
        sum_map = sum_map.view(B, 1, G, G)
        mean_map = sum_map / count_map.clamp_min(1.0)
        count_map = count_map / count_map.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return count_map, mean_map

    def rasterize_bs(self, bs_xy):
        B = bs_xy.shape[0]
        G = self.grid_size
        bs = bs_xy[:, 0]
        x = torch.round(bs[:, 0] * (G - 1)).long().clamp(0, G - 1)
        y = torch.round(bs[:, 1] * (G - 1)).long().clamp(0, G - 1)

        bs_map = torch.zeros(B, 1, G, G, device=bs_xy.device, dtype=bs_xy.dtype)
        batch_idx = torch.arange(B, device=bs_xy.device)
        bs_map[batch_idx, 0, y, x] = 1.0

        dist = torch.sqrt((self.grid_x.unsqueeze(0) - bs[:, 0].view(B, 1, 1)) ** 2 + (self.grid_y.unsqueeze(0) - bs[:, 1].view(B, 1, 1)) ** 2)
        dist = dist.unsqueeze(1)
        dist = dist / dist.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return bs_map, dist

    def build_inputs(self, city, meas_xy, meas_v, bs_xy):
        city_map = F.interpolate(city.unsqueeze(1).float(), size=(self.grid_size, self.grid_size), mode="nearest")
        meas_count, meas_mean = self.rasterize_measurements(meas_xy, meas_v)
        bs_map, bs_dist = self.rasterize_bs(bs_xy)
        return torch.cat([city_map, meas_count, meas_mean, bs_map, bs_dist], dim=1)

    def forward(self, meas_xy, meas_v, bs_xy, task_id=None, city=None, return_debug=False):
        if city is None:
            raise ValueError("SanitySoftBaseline requires city input.")
        x = self.build_inputs(city, meas_xy, meas_v, bs_xy)
        feat0 = self.encoder[0](x)
        feat0 = self.encoder[1](feat0)
        feat1 = self.encoder[2](feat0)
        feat1 = self.encoder[3](feat1)
        feat = self.encoder[4](feat1)
        feat = self.encoder[5](feat)
        logits = self.coverage_hole_head(feat).squeeze(1)
        if return_debug:
            debug = {
                "input_grid": x,
                "feat0": feat0,
                "feat1": feat1,
                "pre_head": feat,
            }
            return logits, debug
        return logits

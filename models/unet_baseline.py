import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)
class UNetBaseline(nn.Module):
    """
    Minimal 2D U-Net baseline that uses only the generic cached single-BS inputs:
    city mask, rasterized measurements, and BS location cues.

    The model predicts a single-channel logit map at the same grid resolution as
    the training target and avoids depending on TORM-G-specific model blocks.
    """

    def __init__(self, grid_size=64, base_channels=32):
        super().__init__()
        self.grid_size = grid_size
        self.base_channels = base_channels

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", gx, persistent=False)
        self.register_buffer("grid_y", gy, persistent=False)

        in_channels = 5
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels)

        self.coverage_hole_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def rasterize_measurements(self, meas_xy, meas_v):
        batch_size, _, _ = meas_xy.shape
        grid_size = self.grid_size
        x = torch.round(meas_xy[..., 0] * (grid_size - 1)).long().clamp(0, grid_size - 1)
        y = torch.round(meas_xy[..., 1] * (grid_size - 1)).long().clamp(0, grid_size - 1)
        flat_idx = y * grid_size + x

        count_map = torch.zeros(batch_size, grid_size * grid_size, device=meas_xy.device, dtype=meas_v.dtype)
        sum_map = torch.zeros(batch_size, grid_size * grid_size, device=meas_xy.device, dtype=meas_v.dtype)
        count_map.scatter_add_(1, flat_idx, torch.ones_like(meas_v))
        sum_map.scatter_add_(1, flat_idx, meas_v)

        count_map = count_map.view(batch_size, 1, grid_size, grid_size)
        sum_map = sum_map.view(batch_size, 1, grid_size, grid_size)
        mean_map = sum_map / count_map.clamp_min(1.0)
        count_map = count_map / count_map.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return count_map, mean_map

    def rasterize_bs(self, bs_xy):
        batch_size = bs_xy.shape[0]
        grid_size = self.grid_size
        bs = bs_xy[:, 0]
        x = torch.round(bs[:, 0] * (grid_size - 1)).long().clamp(0, grid_size - 1)
        y = torch.round(bs[:, 1] * (grid_size - 1)).long().clamp(0, grid_size - 1)

        bs_map = torch.zeros(batch_size, 1, grid_size, grid_size, device=bs_xy.device, dtype=bs_xy.dtype)
        batch_idx = torch.arange(batch_size, device=bs_xy.device)
        bs_map[batch_idx, 0, y, x] = 1.0

        dist = torch.sqrt(
            (self.grid_x.unsqueeze(0) - bs[:, 0].view(batch_size, 1, 1)) ** 2
            + (self.grid_y.unsqueeze(0) - bs[:, 1].view(batch_size, 1, 1)) ** 2
        )
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
            raise ValueError("UNetBaseline requires city input.")

        x = self.build_inputs(city, meas_xy, meas_v, bs_xy)
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec1 = self.up1(bottleneck)
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.dec1(dec1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)

        logits = self.coverage_hole_head(dec2).squeeze(1)
        if return_debug:
            debug = {
                "input_grid": x,
                "enc1": enc1,
                "enc2": enc2,
                "bottleneck": bottleneck,
                "pre_head": dec2,
            }
            return logits, debug
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = ConvBNGELU(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = ConvBNGELU(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LiteASPP(nn.Module):
    def __init__(self, channels, atrous_rates=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                )
                for rate in atrous_rates
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(atrous_rates), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3LiteBaseline(nn.Module):
    """
    Lightweight DeepLabV3-style direct CHD baseline.

    Important: this baseline reuses exactly the same 5-channel direct input
    construction as UNetBaseline / FCNBaseline:
    city, meas_count, meas_mean, bs_map, bs_dist.
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

        c = base_channels
        self.stem = ConvBNGELU(5, c, kernel_size=3, stride=1, padding=1)

        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(c, c * 2, stride=2),
            DepthwiseSeparableConv(c * 2, c * 2, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(c * 2, c * 4, stride=2),
            DepthwiseSeparableConv(c * 4, c * 4, stride=1),
        )

        self.aspp = LiteASPP(c * 4, atrous_rates=(1, 3, 5))
        self.fuse_low = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            ConvBNGELU(c * 5, c * 2, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(c * 2, c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        self.coverage_hole_head = self.decoder[-1]

    # Kept identical to UNetBaseline/FCNBaseline for fairness.
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

    # Kept identical to UNetBaseline/FCNBaseline for fairness.
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
            raise ValueError("DeepLabV3LiteBaseline requires city input.")

        x = self.build_inputs(city, meas_xy, meas_v, bs_xy)
        feat0 = self.stem(x)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)

        aspp_feat = self.aspp(feat2)
        aspp_up = F.interpolate(aspp_feat, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        low = self.fuse_low(feat1)
        fused = torch.cat([aspp_up, low], dim=1)
        logits_low = self.decoder(fused)
        logits = F.interpolate(logits_low, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        logits = logits.squeeze(1)

        if return_debug:
            debug = {
                "input_grid": x,
                "feat0": feat0,
                "feat1": feat1,
                "feat2": feat2,
                "aspp": aspp_feat,
            }
            return logits, debug
        return logits

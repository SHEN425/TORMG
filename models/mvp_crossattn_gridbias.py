import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mvp_crossattn import CrossAttnBlock


class MVP_CrossAttn_GridBias(nn.Module):
    """
    Minimal variant of MVP_CrossAttn that injects a shallow 2D grid bias into
    the latent grid before cross-attention.

    The bias is built from a simplified rasterized 2D view of the current
    inputs:
    - city mask
    - measurement count
    - measurement mean value
    - BS one-hot location
    - BS distance map
    """

    def __init__(self, d=128, heads=4, layers=2, grid_size=64, num_tasks=1, patch_size=9, gridbias_channels=32):
        super().__init__()
        self.d = d
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        self.patch_size = patch_size
        self.gridbias_channels = gridbias_channels
        K = grid_size * grid_size

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([gx, gy], dim=-1).view(K, 2)
        self.register_buffer("grid_xy", grid_xy, persistent=False)
        self.register_buffer("grid_x", gx, persistent=False)
        self.register_buffer("grid_y", gy, persistent=False)

        self.meas_proj = nn.Linear(3, d)
        self.city_patch_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.bs_proj = nn.Linear(2, d)
        self.task_embed = nn.Embedding(num_tasks, d)
        self.task_proj = nn.Linear(d, d)
        self.grid_pos = nn.Parameter(torch.randn(K, d) * 0.02)

        self.grid_bias_stem = nn.Sequential(
            nn.Conv2d(5, gridbias_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(gridbias_channels, d, kernel_size=3, padding=1),
        )

        self.blocks = nn.ModuleList([CrossAttnBlock(d=d, heads=heads, dropout=0.1) for _ in range(layers)])

        self.task_feat_proj_weight = nn.Parameter(torch.randn(num_tasks, d, d) * 0.02)
        self.task_feat_proj_bias = nn.Parameter(torch.zeros(num_tasks, d))
        self.coverage_hole_head = nn.Linear(d, 1)

    def apply_task_projection(self, x, task_id):
        weight = self.task_feat_proj_weight[task_id]
        bias = self.task_feat_proj_bias[task_id]
        x = torch.einsum("bkd,bdh->bkh", x, weight)
        x = x + bias.unsqueeze(1)
        return F.gelu(x)

    def extract_city_patches(self, city, meas_xy):
        B, H, W = city.shape
        _, N, _ = meas_xy.shape
        radius = self.patch_size // 2

        x = torch.round(meas_xy[..., 0] * (W - 1)).long().clamp(0, W - 1)
        y = torch.round(meas_xy[..., 1] * (H - 1)).long().clamp(0, H - 1)

        city_pad = F.pad(city.unsqueeze(1), (radius, radius, radius, radius), mode="constant", value=0.0).squeeze(1)

        offsets = torch.arange(-radius, radius + 1, device=city.device, dtype=torch.long)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dy = dy.view(1, 1, self.patch_size, self.patch_size)
        dx = dx.view(1, 1, self.patch_size, self.patch_size)

        y0 = y.unsqueeze(-1).unsqueeze(-1) + radius
        x0 = x.unsqueeze(-1).unsqueeze(-1) + radius
        yy = y0 + dy
        xx = x0 + dx

        batch_idx = torch.arange(B, device=city.device, dtype=torch.long).view(B, 1, 1, 1).expand(B, N, self.patch_size, self.patch_size)
        return city_pad[batch_idx, yy, xx]

    def encode_city_patches(self, city, meas_xy):
        if city is None:
            B, N, _ = meas_xy.shape
            return torch.zeros(B, N, self.d, dtype=meas_xy.dtype, device=meas_xy.device)
        patches = self.extract_city_patches(city.float(), meas_xy)
        B, N, _, _ = patches.shape
        feat = self.city_patch_encoder(patches.view(B * N, self.patch_size, self.patch_size))
        return feat.view(B, N, self.d)

    def build_distance_bias(self, src_xy, batch_size):
        grid_xy = self.grid_xy.unsqueeze(0).expand(batch_size, -1, -1)
        dist = torch.linalg.norm(grid_xy.unsqueeze(2) - src_xy.unsqueeze(1), dim=-1)
        return -dist

    def build_measurement_density(self, meas_xy, sigma=0.08):
        grid_xy = self.grid_xy.unsqueeze(0).expand(meas_xy.shape[0], -1, -1)
        diff = grid_xy.unsqueeze(2) - meas_xy.unsqueeze(1)
        dist2 = (diff * diff).sum(dim=-1)
        density = torch.exp(-dist2 / (2.0 * sigma * sigma)).mean(dim=-1)
        return density

    def rasterize_measurements(self, meas_xy, meas_v):
        B, _, _ = meas_xy.shape
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

    def build_grid_bias(self, city, meas_xy, meas_v, bs_xy):
        city_map = F.interpolate(city.unsqueeze(1).float(), size=(self.grid_size, self.grid_size), mode="nearest")
        meas_count, meas_mean = self.rasterize_measurements(meas_xy, meas_v)
        bs_map, bs_dist = self.rasterize_bs(bs_xy)
        grid_inputs = torch.cat([city_map, meas_count, meas_mean, bs_map, bs_dist], dim=1)
        grid_bias = self.grid_bias_stem(grid_inputs)
        grid_bias = grid_bias.permute(0, 2, 3, 1).contiguous().view(city.shape[0], self.grid_size * self.grid_size, self.d)
        return grid_inputs, grid_bias

    def forward(self, meas_xy, meas_v, bs_xy, task_id=None, city=None, return_debug=False):
        B, _, _ = meas_xy.shape
        K = self.grid_size * self.grid_size

        meas_in = torch.cat([meas_xy, meas_v.unsqueeze(-1)], dim=-1)
        meas_tok = self.meas_proj(meas_in)
        meas_tok = meas_tok + self.encode_city_patches(city, meas_xy)
        bs_tok = self.bs_proj(bs_xy)
        src = torch.cat([meas_tok, bs_tok], dim=1)
        src_xy = torch.cat([meas_xy, bs_xy], dim=1)
        attn_bias = self.build_distance_bias(src_xy, B)
        density = self.build_measurement_density(meas_xy)

        grid = self.grid_pos.unsqueeze(0).expand(B, K, self.d)
        if task_id is None:
            task_id = torch.zeros(B, dtype=torch.long, device=meas_xy.device)
        task_vec = self.task_proj(self.task_embed(task_id)).unsqueeze(1)
        grid_inputs, grid_bias = self.build_grid_bias(city, meas_xy, meas_v, bs_xy)
        grid = grid + task_vec + grid_bias

        debug = None
        if return_debug:
            debug = {
                "meas_tok": meas_tok,
                "src_tok": src,
                "grid_bias": grid_bias,
                "grid_init": grid,
            }

        for block_idx, blk in enumerate(self.blocks):
            grid = blk(grid, src, attn_bias=attn_bias, density=density)
            if return_debug and block_idx == 0:
                debug["grid_after_block0"] = grid

        grid_out = grid
        task_feat = self.apply_task_projection(grid_out, task_id)
        logits = self.coverage_hole_head(task_feat).squeeze(-1).view(B, self.grid_size, self.grid_size)
        if return_debug:
            debug["grid_out"] = grid_out
            debug["task_feat"] = task_feat
            debug["grid_inputs"] = grid_inputs
            return logits, debug
        return logits

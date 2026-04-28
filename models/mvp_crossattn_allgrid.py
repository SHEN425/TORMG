import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mvp_crossattn import CrossAttnBlock


class MVP_CrossAttn_AllGrid(nn.Module):
    """
    Paper-aligned model with unified grid grounding:
    - measurement branch: tokens + grid-aligned measurement map encoding
    - BS branch: tokens + grid-aligned BS map encoding
    - environment branch: city mask -> environment grid encoding

    The grounded 2D priors from all three branches are fused into a shared
    latent-grid prior before cross-attention. Measurement/BS tokens are kept as
    auxiliary sources for token-to-grid interaction.
    """

    def __init__(
        self,
        d=128,
        heads=4,
        layers=2,
        grid_size=64,
        num_tasks=1,
        meas_grid_channels=32,
        bs_grid_channels=32,
        env_channels=32,
    ):
        super().__init__()
        self.d = d
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        k = grid_size * grid_size

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([gx, gy], dim=-1).view(k, 2)
        self.register_buffer("grid_xy", grid_xy, persistent=False)
        self.register_buffer("grid_x", gx, persistent=False)
        self.register_buffer("grid_y", gy, persistent=False)

        self.meas_proj = nn.Sequential(
            nn.Linear(3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.bs_proj = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.meas_grid_encoder = nn.Sequential(
            nn.Conv2d(2, meas_grid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(meas_grid_channels, d, kernel_size=3, padding=1),
        )
        self.bs_grid_encoder = nn.Sequential(
            nn.Conv2d(2, bs_grid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(bs_grid_channels, d, kernel_size=3, padding=1),
        )
        self.env_encoder = nn.Sequential(
            nn.Conv2d(1, env_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(env_channels, env_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(env_channels, d, kernel_size=1),
        )
        self.grid_fuse = nn.Sequential(
            nn.Conv2d(3 * d, d, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=1),
        )

        self.meas_scale_raw = nn.Parameter(torch.zeros(1))
        self.bs_scale_raw = nn.Parameter(torch.zeros(1))
        self.env_scale_raw = nn.Parameter(torch.zeros(1))
        self.block0_scale_raw = nn.Parameter(torch.zeros(1))

        self.task_embed = nn.Embedding(num_tasks, d)
        self.task_proj = nn.Linear(d, d)
        self.grid_pos = nn.Parameter(torch.randn(k, d) * 0.02)

        self.blocks = nn.ModuleList([CrossAttnBlock(d=d, heads=heads, dropout=0.1) for _ in range(layers)])

        self.task_feat_proj_weight = nn.Parameter(torch.randn(num_tasks, d, d) * 0.02)
        self.task_feat_proj_bias = nn.Parameter(torch.zeros(num_tasks, d))
        self.coverage_hole_head = nn.Linear(d, 1)

    def ensure_bs_tokens(self, bs_xy):
        if bs_xy.ndim == 2:
            return bs_xy.unsqueeze(1)
        return bs_xy

    def apply_task_projection(self, x, task_id):
        weight = self.task_feat_proj_weight[task_id]
        bias = self.task_feat_proj_bias[task_id]
        x = torch.einsum("bkd,bdh->bkh", x, weight)
        x = x + bias.unsqueeze(1)
        return F.gelu(x)

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
        b, _, _ = meas_xy.shape
        g = self.grid_size
        x = torch.round(meas_xy[..., 0] * (g - 1)).long().clamp(0, g - 1)
        y = torch.round(meas_xy[..., 1] * (g - 1)).long().clamp(0, g - 1)
        flat_idx = y * g + x

        count_map = torch.zeros(b, g * g, device=meas_xy.device, dtype=meas_v.dtype)
        sum_map = torch.zeros(b, g * g, device=meas_xy.device, dtype=meas_v.dtype)
        count_map.scatter_add_(1, flat_idx, torch.ones_like(meas_v))
        sum_map.scatter_add_(1, flat_idx, meas_v)
        count_map = count_map.view(b, 1, g, g)
        sum_map = sum_map.view(b, 1, g, g)
        mean_map = sum_map / count_map.clamp_min(1.0)
        count_map = count_map / count_map.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return count_map, mean_map

    def rasterize_bs(self, bs_xy):
        b, m, _ = bs_xy.shape
        g = self.grid_size

        x = torch.round(bs_xy[..., 0] * (g - 1)).long().clamp(0, g - 1)
        y = torch.round(bs_xy[..., 1] * (g - 1)).long().clamp(0, g - 1)

        bs_map = torch.zeros(b, 1, g, g, device=bs_xy.device, dtype=bs_xy.dtype)
        batch_idx = torch.arange(b, device=bs_xy.device).view(b, 1).expand(b, m)
        bs_map[batch_idx, 0, y, x] += 1.0
        bs_map = bs_map / bs_map.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)

        gx = self.grid_x.unsqueeze(0).unsqueeze(1)
        gy = self.grid_y.unsqueeze(0).unsqueeze(1)
        bx = bs_xy[..., 0].view(b, m, 1, 1)
        by = bs_xy[..., 1].view(b, m, 1, 1)
        dist = torch.sqrt((gx - bx) ** 2 + (gy - by) ** 2)
        dist = dist.min(dim=1).values.unsqueeze(1)
        dist = dist / dist.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return bs_map, dist

    def encode_environment_grid(self, city):
        if city is None:
            raise ValueError("MVP_CrossAttn_AllGrid requires city input.")
        city_grid = F.interpolate(city.unsqueeze(1).float(), size=(self.grid_size, self.grid_size), mode="nearest")
        return self.env_encoder(city_grid)

    def build_grounded_grids(self, city, meas_xy, meas_v, bs_xy):
        meas_count, meas_mean = self.rasterize_measurements(meas_xy, meas_v)
        meas_grid = self.meas_grid_encoder(torch.cat([meas_count, meas_mean], dim=1))

        bs_map, bs_dist = self.rasterize_bs(bs_xy)
        bs_grid = self.bs_grid_encoder(torch.cat([bs_map, bs_dist], dim=1))

        env_grid = self.encode_environment_grid(city)

        meas_scale = F.softplus(self.meas_scale_raw)
        bs_scale = F.softplus(self.bs_scale_raw)
        env_scale = F.softplus(self.env_scale_raw)
        fused = self.grid_fuse(
            torch.cat(
                [
                    meas_scale * meas_grid,
                    bs_scale * bs_grid,
                    env_scale * env_grid,
                ],
                dim=1,
            )
        )
        return meas_grid, bs_grid, env_grid, fused, meas_scale, bs_scale, env_scale

    def forward(self, meas_xy, meas_v, bs_xy, task_id=None, city=None, return_debug=False):
        b, _, _ = meas_xy.shape
        k = self.grid_size * self.grid_size
        bs_xy = self.ensure_bs_tokens(bs_xy)

        meas_in = torch.cat([meas_xy, meas_v.unsqueeze(-1)], dim=-1)
        meas_tok = self.meas_proj(meas_in)
        bs_tok = self.bs_proj(bs_xy)
        src = torch.cat([meas_tok, bs_tok], dim=1)
        src_xy = torch.cat([meas_xy, bs_xy], dim=1)
        attn_bias = self.build_distance_bias(src_xy, b)
        density = self.build_measurement_density(meas_xy)

        meas_grid, bs_grid, env_grid, grounded_grid_map, meas_scale, bs_scale, env_scale = self.build_grounded_grids(
            city, meas_xy, meas_v, bs_xy
        )
        grounded_grid = grounded_grid_map.permute(0, 2, 3, 1).contiguous().view(b, k, self.d)

        grid = self.grid_pos.unsqueeze(0).expand(b, k, self.d)
        if task_id is None:
            task_id = torch.zeros(b, dtype=torch.long, device=meas_xy.device)
        task_vec = self.task_proj(self.task_embed(task_id)).unsqueeze(1)
        grid = grid + task_vec + grounded_grid

        block0_scale = F.softplus(self.block0_scale_raw)

        debug = None
        if return_debug:
            debug = {
                "meas_tok": meas_tok,
                "bs_tok": bs_tok,
                "src_tok": src,
                "meas_grid": meas_grid,
                "bs_grid": bs_grid,
                "env_grid": env_grid,
                "grounded_grid": grounded_grid_map,
                "meas_scale": meas_scale.view(1),
                "bs_scale": bs_scale.view(1),
                "env_scale": env_scale.view(1),
                "block0_scale": block0_scale.view(1),
                "grid_init": grid,
            }

        for block_idx, blk in enumerate(self.blocks):
            grid = blk(grid, src, attn_bias=attn_bias, density=density)
            if block_idx == 0:
                grid = grid + block0_scale * grounded_grid
                if return_debug:
                    debug["grid_after_block0"] = grid

        grid_out = grid
        task_feat = self.apply_task_projection(grid_out, task_id)
        logits = self.coverage_hole_head(task_feat).squeeze(-1).view(b, self.grid_size, self.grid_size)
        if return_debug:
            debug["grid_out"] = grid_out
            debug["task_feat"] = task_feat
            return logits, debug
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mvp_crossattn import CrossAttnBlock


class MVP_CrossAttn_EnvTokens(nn.Module):
    """
    Paper-aligned variant with three explicit branches:
    - measurement tokens from (x, y, value)
    - BS tokens from BS coordinates, kept variable-length for future multi-BS
    - environment tokens from a shallow CNN over the city/environment mask

    The three token groups are fused through the same latent-grid cross-attention
    path used by the main model.
    """

    def __init__(
        self,
        d=128,
        heads=4,
        layers=2,
        grid_size=64,
        num_tasks=1,
        env_token_grid_size=16,
        env_channels=32,
    ):
        super().__init__()
        self.d = d
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        self.env_token_grid_size = env_token_grid_size
        self.env_channels = env_channels
        K = grid_size * grid_size

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([gx, gy], dim=-1).view(K, 2)
        self.register_buffer("grid_xy", grid_xy, persistent=False)

        env_ys = (torch.arange(env_token_grid_size, dtype=torch.float32) + 0.5) / env_token_grid_size
        env_xs = (torch.arange(env_token_grid_size, dtype=torch.float32) + 0.5) / env_token_grid_size
        env_gy, env_gx = torch.meshgrid(env_ys, env_xs, indexing="ij")
        env_xy = torch.stack([env_gx, env_gy], dim=-1).view(env_token_grid_size * env_token_grid_size, 2)
        self.register_buffer("env_xy", env_xy, persistent=False)

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

        self.env_encoder = nn.Sequential(
            nn.Conv2d(1, env_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(env_channels, env_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(env_channels, d, kernel_size=1),
        )

        self.task_embed = nn.Embedding(num_tasks, d)
        self.task_proj = nn.Linear(d, d)
        self.grid_pos = nn.Parameter(torch.randn(K, d) * 0.02)

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

    def encode_environment(self, city):
        if city is None:
            raise ValueError("MVP_CrossAttn_EnvTokens requires city input for the environment branch.")
        city_small = F.interpolate(
            city.unsqueeze(1).float(),
            size=(self.env_token_grid_size, self.env_token_grid_size),
            mode="nearest",
        )
        env_feat_map = self.env_encoder(city_small)
        env_tok = env_feat_map.permute(0, 2, 3, 1).contiguous().view(city.shape[0], -1, self.d)
        return env_feat_map, env_tok

    def ensure_bs_tokens(self, bs_xy):
        if bs_xy.ndim == 2:
            return bs_xy.unsqueeze(1)
        return bs_xy

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

    def forward(self, meas_xy, meas_v, bs_xy, task_id=None, city=None, return_debug=False):
        B, _, _ = meas_xy.shape
        K = self.grid_size * self.grid_size

        bs_xy = self.ensure_bs_tokens(bs_xy)
        meas_in = torch.cat([meas_xy, meas_v.unsqueeze(-1)], dim=-1)
        meas_tok = self.meas_proj(meas_in)
        bs_tok = self.bs_proj(bs_xy)
        env_feat_map, env_tok = self.encode_environment(city)

        src = torch.cat([meas_tok, bs_tok, env_tok], dim=1)
        env_xy = self.env_xy.unsqueeze(0).expand(B, -1, -1)
        src_xy = torch.cat([meas_xy, bs_xy, env_xy], dim=1)
        attn_bias = self.build_distance_bias(src_xy, B)
        density = self.build_measurement_density(meas_xy)

        grid = self.grid_pos.unsqueeze(0).expand(B, K, self.d)
        if task_id is None:
            task_id = torch.zeros(B, dtype=torch.long, device=meas_xy.device)
        task_vec = self.task_proj(self.task_embed(task_id)).unsqueeze(1)
        grid = grid + task_vec

        debug = None
        if return_debug:
            debug = {
                "meas_tok": meas_tok,
                "bs_tok": bs_tok,
                "env_tok": env_tok,
                "src_tok": src,
                "env_feat_map": env_feat_map,
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
            return logits, debug
        return logits

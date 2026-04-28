import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mvp_crossattn import CrossAttnBlock


class MVP_CrossAttn_EnvGridPlus(nn.Module):
    """
    Minimal strengthened version of MVP_CrossAttn_EnvGrid.

    Changes relative to envgrid:
    - learnable anchoring strength for environment-to-grid injection
    - a second early injection right after the first cross-attention block
    """

    def __init__(
        self,
        d=128,
        heads=4,
        layers=2,
        grid_size=64,
        num_tasks=1,
        env_channels=32,
        enable_aux_rm_head=False,
        aux_rm_detach_from_chd=False,
        enable_obstruction_bias=False,
        obstruction_num_samples=16,
        use_refine_head=False,
        refine_channels=32,
        refine_layers=2,
        refine_scale=1.0,
    ):
        super().__init__()
        self.d = d
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        self.enable_aux_rm_head = bool(enable_aux_rm_head)
        self.aux_rm_detach_from_chd = bool(aux_rm_detach_from_chd)
        self.enable_obstruction_bias = bool(enable_obstruction_bias)
        self.obstruction_num_samples = int(max(2, obstruction_num_samples))
        self.use_refine_head = bool(use_refine_head)
        self.refine_scale = float(refine_scale)
        K = grid_size * grid_size

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([gx, gy], dim=-1).view(K, 2)
        self.register_buffer("grid_xy", grid_xy, persistent=False)

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
        self.env_grid_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.env_init_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.env_block0_scale_raw = nn.Parameter(torch.tensor(0.0))

        self.task_embed = nn.Embedding(num_tasks, d)
        self.task_proj = nn.Linear(d, d)
        self.grid_pos = nn.Parameter(torch.randn(K, d) * 0.02)

        self.blocks = nn.ModuleList([CrossAttnBlock(d=d, heads=heads, dropout=0.1) for _ in range(layers)])

        self.task_feat_proj_weight = nn.Parameter(torch.randn(num_tasks, d, d) * 0.02)
        self.task_feat_proj_bias = nn.Parameter(torch.zeros(num_tasks, d))
        self.coverage_hole_head = nn.Linear(d, 1)
        self.refine_head = None
        if self.use_refine_head:
            ch = int(max(4, refine_channels))
            n_layers = int(max(2, refine_layers))
            refine_blocks = []
            in_ch = d
            for _ in range(n_layers):
                refine_blocks.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
                refine_blocks.append(nn.GELU())
                in_ch = ch
            refine_blocks.append(nn.Conv2d(ch, 1, kernel_size=3, padding=1))
            self.refine_head = nn.Sequential(*refine_blocks)
        # Optional RM-oriented auxiliary head. Disabled by default to preserve
        # baseline behavior.
        self.aux_rm_head = nn.Linear(d, 1) if self.enable_aux_rm_head else None

    def apply_task_projection(self, x, task_id):
        weight = self.task_feat_proj_weight[task_id]
        bias = self.task_feat_proj_bias[task_id]
        x = torch.einsum("bkd,bdh->bkh", x, weight)
        x = x + bias.unsqueeze(1)
        return F.gelu(x)

    def ensure_bs_tokens(self, bs_xy):
        if bs_xy.ndim == 2:
            return bs_xy.unsqueeze(1)
        return bs_xy

    def encode_environment(self, city):
        if city is None:
            raise ValueError("MVP_CrossAttn_EnvGridPlus requires city input for the environment branch.")
        city_grid = F.interpolate(
            city.unsqueeze(1).float(),
            size=(self.grid_size, self.grid_size),
            mode="nearest",
        )
        env_feat_map = self.env_encoder(city_grid)
        env_grid = env_feat_map.permute(0, 2, 3, 1).contiguous().view(city.shape[0], -1, self.d)
        env_grid = self.env_grid_proj(env_grid)
        return env_feat_map, env_grid

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

    def build_obstruction_bias(self, city, meas_xy, bs_xy):
        """
        Build obstruction-aware bias for attention.

        Minimal version: compute line-of-sight blockage ratio from each latent
        grid cell to BS tokens and apply it only on BS columns of the source
        sequence. Measurement-token columns are left untouched.
        """
        b, n, _ = meas_xy.shape
        m = bs_xy.shape[1]
        k = self.grid_size * self.grid_size
        s = n + m

        city_grid = F.interpolate(city.unsqueeze(1).float(), size=(self.grid_size, self.grid_size), mode="nearest").squeeze(1)
        city_flat = city_grid.view(b, -1)
        grid_xy = self.grid_xy.unsqueeze(0).expand(b, -1, -1)

        # Exclude the endpoints (grid cell and BS cell) for robust blockage ratio.
        alphas = torch.linspace(0.0, 1.0, steps=self.obstruction_num_samples, device=city.device)[1:-1]
        obstruction_sum = torch.zeros(b, k, m, device=city.device, dtype=city.dtype)

        for alpha in alphas:
            line_xy = (1.0 - alpha) * grid_xy.unsqueeze(2) + alpha * bs_xy.unsqueeze(1)
            x = torch.round(line_xy[..., 0] * (self.grid_size - 1)).long().clamp(0, self.grid_size - 1)
            y = torch.round(line_xy[..., 1] * (self.grid_size - 1)).long().clamp(0, self.grid_size - 1)
            flat_idx = (y * self.grid_size + x).view(b, -1)
            sampled = city_flat.gather(1, flat_idx).view(b, k, m)
            obstruction_sum = obstruction_sum + sampled

        denom = float(max(len(alphas), 1))
        obstruction_ratio = (obstruction_sum / denom).clamp(0.0, 1.0)

        obs_bias = torch.zeros(b, k, s, device=city.device, dtype=city.dtype)
        obs_bias[:, :, n:] = -obstruction_ratio
        return obs_bias, obstruction_ratio

    def forward(self, meas_xy, meas_v, bs_xy, task_id=None, city=None, return_debug=False, return_aux=False):
        B, N, _ = meas_xy.shape
        K = self.grid_size * self.grid_size

        bs_xy = self.ensure_bs_tokens(bs_xy)
        meas_in = torch.cat([meas_xy, meas_v.unsqueeze(-1)], dim=-1)
        meas_tok = self.meas_proj(meas_in)
        bs_tok = self.bs_proj(bs_xy)
        src = torch.cat([meas_tok, bs_tok], dim=1)
        src_xy = torch.cat([meas_xy, bs_xy], dim=1)
        attn_bias = self.build_distance_bias(src_xy, B)
        obs_bias = None
        obstruction_ratio = None
        if self.enable_obstruction_bias:
            obs_bias, obstruction_ratio = self.build_obstruction_bias(city, meas_xy, bs_xy)
        density = self.build_measurement_density(meas_xy)

        env_feat_map, env_grid = self.encode_environment(city)
        env_init_scale = F.softplus(self.env_init_scale_raw)
        env_block0_scale = F.softplus(self.env_block0_scale_raw)

        grid = self.grid_pos.unsqueeze(0).expand(B, K, self.d)
        if task_id is None:
            task_id = torch.zeros(B, dtype=torch.long, device=meas_xy.device)
        task_vec = self.task_proj(self.task_embed(task_id)).unsqueeze(1)
        grid = grid + task_vec + env_init_scale * env_grid

        debug = None
        if return_debug:
            debug = {
                "meas_tok": meas_tok,
                "bs_tok": bs_tok,
                "src_tok": src,
                "env_feat_map": env_feat_map,
                "env_grid": env_grid,
                "grid_init": grid,
                "env_init_scale": env_init_scale.view(1, 1),
                "env_block0_scale": env_block0_scale.view(1, 1),
            }
            if obstruction_ratio is not None:
                debug["obstruction_ratio"] = obstruction_ratio

        for block_idx, blk in enumerate(self.blocks):
            grid = blk(grid, src, attn_bias=attn_bias, obs_bias=obs_bias, density=density)
            if block_idx == 0:
                grid = grid + env_block0_scale * env_grid
                if return_debug:
                    debug["grid_after_block0"] = grid

        grid_out = grid
        task_feat = self.apply_task_projection(grid_out, task_id)
        logits_base = self.coverage_hole_head(task_feat).squeeze(-1).view(B, self.grid_size, self.grid_size)
        logits_delta = None
        logits = logits_base
        if self.refine_head is not None:
            task_feat_map = task_feat.view(B, self.grid_size, self.grid_size, self.d).permute(0, 3, 1, 2).contiguous()
            logits_delta = self.refine_head(task_feat_map).squeeze(1)
            logits = logits_base + self.refine_scale * logits_delta

        aux_rm_logits = None
        if self.aux_rm_head is not None:
            # Optional detach to avoid auxiliary RM supervision dominating the
            # CHD backbone updates.
            aux_feat = grid_out.detach() if self.aux_rm_detach_from_chd else grid_out
            aux_rm_logits = self.aux_rm_head(aux_feat).squeeze(-1).view(B, self.grid_size, self.grid_size)

        if return_debug:
            debug["grid_out"] = grid_out
            debug["task_feat"] = task_feat
            debug["logits_base"] = logits_base
            if logits_delta is not None:
                debug["logits_delta"] = logits_delta
            if aux_rm_logits is not None:
                debug["aux_rm_logits"] = aux_rm_logits
            if return_aux:
                return logits, debug, aux_rm_logits
            return logits, debug

        if return_aux:
            return logits, aux_rm_logits
        return logits

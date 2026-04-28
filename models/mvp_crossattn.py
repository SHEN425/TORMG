import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads})")
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.dist_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.obstruction_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.density_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, attn_bias=None, obs_bias=None, density=None):
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)

        B, K, _ = q_in.shape
        _, S, _ = kv_in.shape

        q_proj = self.q_proj(q_in).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(kv_in).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(kv_in).view(B, S, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            dist_scale = F.softplus(self.dist_scale_raw)
            attn_scores = attn_scores + dist_scale * attn_bias.unsqueeze(1)
        if obs_bias is not None:
            obstruction_scale = F.softplus(self.obstruction_scale_raw)
            attn_scores = attn_scores + obstruction_scale * obs_bias.unsqueeze(1)
        if density is not None:
            density_scale = torch.tanh(self.density_scale_raw)
            density_gate = 1.0 + density_scale * density.unsqueeze(1).unsqueeze(-1)
            attn_scores = attn_scores * density_gate

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v_proj)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, self.d)
        attn_out = self.out_proj(attn_out)

        q = q + self.dropout(attn_out)
        q = q + self.dropout(self.ffn(self.norm_ffn(q)))
        return q


class MVP_CrossAttn(nn.Module):
    """
    最小 MVP：用 token 化的测量点 + BS 信息，预测 64x64 的 hole。
    实现方式：把 [grid_tokens; src_tokens] 拼接后做 self-attn，然后取 grid 输出。
    """
    def __init__(self, d=128, heads=4, layers=2, grid_size=64, num_tasks=1, patch_size=9):
        super().__init__()
        self.d = d
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        self.patch_size = patch_size
        K = grid_size * grid_size

        ys = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        xs = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([gx, gy], dim=-1).view(K, 2)
        self.register_buffer("grid_xy", grid_xy, persistent=False)

        # meas token: (x,y,v) -> d
        self.meas_proj = nn.Linear(3, d)
        self.city_patch_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        # bs token: (x,y) -> d
        self.bs_proj = nn.Linear(2, d)
        # task embedding -> latent modulation
        self.task_embed = nn.Embedding(num_tasks, d)
        self.task_proj = nn.Linear(d, d)

        # grid positional embedding: (K,d)
        self.grid_pos = nn.Parameter(torch.randn(K, d) * 0.02)

        self.blocks = nn.ModuleList([CrossAttnBlock(d=d, heads=heads, dropout=0.1) for _ in range(layers)])

        # Task-specific latent projection before the task head.
        self.task_feat_proj_weight = nn.Parameter(torch.randn(num_tasks, d, d) * 0.02)
        self.task_feat_proj_bias = nn.Parameter(torch.zeros(num_tasks, d))

        # Current active task head: coverage-hole detection.
        self.coverage_hole_head = nn.Linear(d, 1)

    def apply_task_projection(self, x, task_id):
        """
        x: (B,K,d)
        task_id: (B,)
        """
        weight = self.task_feat_proj_weight[task_id]   # (B,d,d)
        bias = self.task_feat_proj_bias[task_id]       # (B,d)
        x = torch.einsum("bkd,bdh->bkh", x, weight)
        x = x + bias.unsqueeze(1)
        return F.gelu(x)

    def extract_city_patches(self, city, meas_xy):
        """
        city: (B,H,W)
        meas_xy: (B,N,2) normalized to [0,1]
        return: (B,N,P,P)
        """
        B, H, W = city.shape
        _, N, _ = meas_xy.shape
        radius = self.patch_size // 2

        x = torch.round(meas_xy[..., 0] * (W - 1)).long()
        y = torch.round(meas_xy[..., 1] * (H - 1)).long()
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)

        city_pad = F.pad(city.unsqueeze(1), (radius, radius, radius, radius), mode="constant", value=0.0)
        city_pad = city_pad.squeeze(1)

        offsets = torch.arange(-radius, radius + 1, device=city.device, dtype=torch.long)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dy = dy.view(1, 1, self.patch_size, self.patch_size)
        dx = dx.view(1, 1, self.patch_size, self.patch_size)

        y0 = y.unsqueeze(-1).unsqueeze(-1) + radius
        x0 = x.unsqueeze(-1).unsqueeze(-1) + radius

        yy = y0 + dy
        xx = x0 + dx

        batch_idx = torch.arange(B, device=city.device, dtype=torch.long).view(B, 1, 1, 1).expand(B, N, self.patch_size, self.patch_size)
        patches = city_pad[batch_idx, yy, xx]
        return patches

    def encode_city_patches(self, city, meas_xy):
        if city is None:
            B, N, _ = meas_xy.shape
            return torch.zeros(B, N, self.d, dtype=meas_xy.dtype, device=meas_xy.device)

        patches = self.extract_city_patches(city.float(), meas_xy)
        B, N, _, _ = patches.shape
        feat = self.city_patch_encoder(patches.view(B * N, self.patch_size, self.patch_size))
        return feat.view(B, N, self.d)

    def build_distance_bias(self, src_xy, batch_size):
        """
        src_xy: (B,S,2) normalized to [0,1]
        return: (B,K,S), additive bias before softmax
        """
        grid_xy = self.grid_xy.unsqueeze(0).expand(batch_size, -1, -1)  # (B,K,2)
        dist = torch.linalg.norm(grid_xy.unsqueeze(2) - src_xy.unsqueeze(1), dim=-1)
        return -dist

    def build_measurement_density(self, meas_xy, sigma=0.08):
        """
        meas_xy: (B,N,2) normalized to [0,1]
        return: (B,K), local density estimate per latent cell
        """
        grid_xy = self.grid_xy.unsqueeze(0).expand(meas_xy.shape[0], -1, -1)  # (B,K,2)
        diff = grid_xy.unsqueeze(2) - meas_xy.unsqueeze(1)                     # (B,K,N,2)
        dist2 = (diff * diff).sum(dim=-1)
        density = torch.exp(-dist2 / (2.0 * sigma * sigma)).mean(dim=-1)
        return density

    def forward(self, meas_xy, meas_v, bs_xy, task_id=None, city=None, return_debug=False):
        """
        meas_xy: (B,N,2) in [0,1]
        meas_v : (B,N)   in [0,1]
        bs_xy  : (B,1,2) in [0,1]
        task_id: (B,) task index, defaults to 0
        city   : (B,H,W) city/building mask
        return logits: (B,64,64)
        """
        B, N, _ = meas_xy.shape
        K = self.grid_size * self.grid_size

        meas_in = torch.cat([meas_xy, meas_v.unsqueeze(-1)], dim=-1)  # (B,N,3)
        meas_tok = self.meas_proj(meas_in)                            # (B,N,d)
        meas_tok = meas_tok + self.encode_city_patches(city, meas_xy) # (B,N,d)
        bs_tok = self.bs_proj(bs_xy)                                  # (B,1,d)
        src = torch.cat([meas_tok, bs_tok], dim=1)                    # (B,N+1,d)
        src_xy = torch.cat([meas_xy, bs_xy], dim=1)                   # (B,N+1,2)
        attn_bias = self.build_distance_bias(src_xy, B)               # (B,K,N+1)
        density = self.build_measurement_density(meas_xy)             # (B,K)

        grid = self.grid_pos.unsqueeze(0).expand(B, K, self.d)        # (B,K,d)
        if task_id is None:
            task_id = torch.zeros(B, dtype=torch.long, device=meas_xy.device)
        task_vec = self.task_proj(self.task_embed(task_id)).unsqueeze(1)  # (B,1,d)
        grid = grid + task_vec
        debug = None
        if return_debug:
            debug = {
                "meas_tok": meas_tok,
                "bs_tok": bs_tok,
                "src_tok": src,
                "grid_init": grid,
            }

        for block_idx, blk in enumerate(self.blocks):
            grid = blk(grid, src, attn_bias=attn_bias, density=density)
            if return_debug and block_idx == 0:
                debug["grid_after_block0"] = grid

        grid_out = grid                                                # (B,K,d)
        task_feat = self.apply_task_projection(grid_out, task_id)     # (B,K,d)
        logits = self.coverage_hole_head(task_feat).squeeze(-1).view(B, self.grid_size, self.grid_size)
        if return_debug:
            debug["grid_out"] = grid_out
            debug["task_feat"] = task_feat
            return logits, debug
        return logits

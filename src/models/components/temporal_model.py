from einops import rearrange
import torch
from torch import nn

from src.models.components.classification_head import ClassificationHead


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class SegmentPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, num_segments: int, seg_length: int):
        super().__init__()
        self.seg_emb = nn.Embedding(num_segments, emb_size)
        self.pos_emb = nn.Embedding(seg_length, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, l, d)
        n, l = x.shape[1], x.shape[2]
        seg_bias = self.seg_emb(torch.arange(n, device=x.device))  # (n, d)
        pos_bias = self.pos_emb(torch.arange(l, device=x.device))  # (l, d)
        return x + seg_bias[None, :, None, :] + pos_bias[None, None, :, :]


class TemporalModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        emb_size: int,
        output_size: int,
        heads: int,
        dim_heads: int,
        depth: int,
        num_segments: int,
        seg_length: int,
        temporal_module: str = "axial",
        dropout: float = 0.0,
        drop_path: float = 0.0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.heads = heads
        self.dim_heads = dim_heads
        self.depth = depth
        self.num_segments = num_segments
        self.seg_length = seg_length
        self.temporal_module = temporal_module
        self.drop_path = drop_path
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand

        self.projection = nn.Linear(self.input_size, self.emb_size)
        if self.temporal_module == "axial":
            from axial_attention import AxialImageTransformer

            self.temporal = AxialImageTransformer(
                dim=self.emb_size,
                depth=self.depth,
                heads=self.heads,
                dim_heads=self.dim_heads,
                reversible=True,
                axial_pos_emb_shape=(self.num_segments, self.seg_length),
            )
        elif self.temporal_module == "bimamba1":
            try:
                # mamba-ssm v1.x
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaSSMBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba_fwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_bwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (b, t, d)
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1 uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )
                    x_n = self.norm(x)
                    y_fwd = self.mamba_fwd(x_n)
                    y_bwd = torch.flip(self.mamba_bwd(torch.flip(x_n, dims=[1])), dims=[1])
                    w = torch.sigmoid(self.alpha)
                    y = w * y_fwd + (1.0 - w) * y_bwd
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaSSMBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_st":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaSTBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm_t = nn.LayerNorm(d_model)
                    self.norm_s = nn.LayerNorm(d_model)
                    self.mamba_t_fwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_t_bwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_s_fwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_s_bwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha_t = nn.Parameter(torch.zeros(1))
                    self.alpha_s = nn.Parameter(torch.zeros(1))
                    self.beta = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (b, n, l, d), n for temporal axis, l for intra-segment axis
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_st uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )

                    b, n, l, _ = x.shape

                    x_t = rearrange(x, "b n l d -> b l n d")
                    xt = rearrange(self.norm_t(x_t), "b l n d -> (b l) n d")
                    yt_fwd = self.mamba_t_fwd(xt)
                    yt_bwd = torch.flip(self.mamba_t_bwd(torch.flip(xt, dims=[1])), dims=[1])
                    wt = torch.sigmoid(self.alpha_t)
                    yt = wt * yt_fwd + (1.0 - wt) * yt_bwd
                    yt = rearrange(yt, "(b l) n d -> b l n d", b=b, l=l)
                    yt = rearrange(yt, "b l n d -> b n l d")

                    xs = rearrange(self.norm_s(x), "b n l d -> (b n) l d")
                    ys_fwd = self.mamba_s_fwd(xs)
                    ys_bwd = torch.flip(self.mamba_s_bwd(torch.flip(xs, dims=[1])), dims=[1])
                    ws = torch.sigmoid(self.alpha_s)
                    ys = ws * ys_fwd + (1.0 - ws) * ys_bwd
                    ys = rearrange(ys, "(b n) l d -> b n l d", b=b, n=n)

                    mix = torch.sigmoid(self.beta)
                    y = mix * yt + (1.0 - mix) * ys
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaSTBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_shared":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaSharedTemporalBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (b, t, d), bidirectional temporal scan with shared weights
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_shared uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )
                    x_n = self.norm(x)
                    y_fwd = self.mamba(x_n)
                    y_bwd = torch.flip(self.mamba(torch.flip(x_n, dims=[1])), dims=[1])
                    w = torch.sigmoid(self.alpha)
                    y = w * y_fwd + (1.0 - w) * y_bwd
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaSharedTemporalBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_s":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaSpatialOnlyBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm_s = nn.LayerNorm(d_model)
                    self.mamba_s_fwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_s_bwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha_s = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (b, n, l, d), only scan on l axis (intra-segment axis)
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_s uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )

                    b, n, _, _ = x.shape
                    xs = rearrange(self.norm_s(x), "b n l d -> (b n) l d")
                    ys_fwd = self.mamba_s_fwd(xs)
                    ys_bwd = torch.flip(self.mamba_s_bwd(torch.flip(xs, dims=[1])), dims=[1])
                    ws = torch.sigmoid(self.alpha_s)
                    ys = ws * ys_fwd + (1.0 - ws) * ys_bwd
                    ys = rearrange(ys, "(b n) l d -> b n l d", b=b, n=n)
                    ys = self.dropout(ys)
                    ys = self.drop_path(ys)
                    return x + ys

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaSpatialOnlyBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_s_shared":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaSpatialOnlySharedBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm_s = nn.LayerNorm(d_model)
                    self.mamba_s = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha_s = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_s_shared uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )

                    b, n, _, _ = x.shape
                    xs = rearrange(self.norm_s(x), "b n l d -> (b n) l d")
                    ys_fwd = self.mamba_s(xs)
                    ys_bwd = torch.flip(self.mamba_s(torch.flip(xs, dims=[1])), dims=[1])
                    ws = torch.sigmoid(self.alpha_s)
                    ys = ws * ys_fwd + (1.0 - ws) * ys_bwd
                    ys = rearrange(ys, "(b n) l d -> b n l d", b=b, n=n)
                    ys = self.dropout(ys)
                    ys = self.drop_path(ys)
                    return x + ys

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaSpatialOnlySharedBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_st_shared":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _SharedMambaSTBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm_t = nn.LayerNorm(d_model)
                    self.norm_s = nn.LayerNorm(d_model)
                    # Shared-weight bidirectional scan per axis:
                    # time axis shares one Mamba for fwd/bwd, space axis shares another.
                    self.mamba_t = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_s = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha_t = nn.Parameter(torch.zeros(1))
                    self.alpha_s = nn.Parameter(torch.zeros(1))
                    self.beta = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (b, n, l, d), shared Mamba is applied on both temporal and spatial scans
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_st_shared uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )

                    b, n, l, _ = x.shape
                    x_t = rearrange(x, "b n l d -> b l n d")
                    xt = rearrange(self.norm_t(x_t), "b l n d -> (b l) n d")
                    yt_fwd = self.mamba_t(xt)
                    yt_bwd = torch.flip(self.mamba_t(torch.flip(xt, dims=[1])), dims=[1])
                    wt = torch.sigmoid(self.alpha_t)
                    yt = wt * yt_fwd + (1.0 - wt) * yt_bwd
                    yt = rearrange(yt, "(b l) n d -> b l n d", b=b, l=l)
                    yt = rearrange(yt, "b l n d -> b n l d")

                    xs = rearrange(self.norm_s(x), "b n l d -> (b n) l d")
                    ys_fwd = self.mamba_s(xs)
                    ys_bwd = torch.flip(self.mamba_s(torch.flip(xs, dims=[1])), dims=[1])
                    ws = torch.sigmoid(self.alpha_s)
                    ys = ws * ys_fwd + (1.0 - ws) * ys_bwd
                    ys = rearrange(ys, "(b n) l d -> b n l d", b=b, n=n)

                    mix = torch.sigmoid(self.beta)
                    y = mix * yt + (1.0 - mix) * ys
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_SharedMambaSTBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_mix_space":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaMixedSpaceBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba_fwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_bwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_mix_space uses official mamba-ssm and requires CUDA tensors."
                        )
                    b, n, l, _ = x.shape
                    x_mix = rearrange(x, "b n l d -> b l n d")
                    x_mix = rearrange(self.norm(x_mix), "b l n d -> b (l n) d")
                    y_fwd = self.mamba_fwd(x_mix)
                    y_bwd = torch.flip(self.mamba_bwd(torch.flip(x_mix, dims=[1])), dims=[1])
                    w = torch.sigmoid(self.alpha)
                    y = w * y_fwd + (1.0 - w) * y_bwd
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    y = rearrange(y, "b (l n) d -> b l n d", l=l, n=n)
                    y = rearrange(y, "b l n d -> b n l d")
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaMixedSpaceBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_mix_space_shared":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaMixedSpaceSharedBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_mix_space_shared uses official mamba-ssm and requires CUDA tensors."
                        )
                    b, n, l, _ = x.shape
                    x_mix = rearrange(x, "b n l d -> b l n d")
                    x_mix = rearrange(self.norm(x_mix), "b l n d -> b (l n) d")
                    y_fwd = self.mamba(x_mix)
                    y_bwd = torch.flip(self.mamba(torch.flip(x_mix, dims=[1])), dims=[1])
                    w = torch.sigmoid(self.alpha)
                    y = w * y_fwd + (1.0 - w) * y_bwd
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    y = rearrange(y, "b (l n) d -> b l n d", l=l, n=n)
                    y = rearrange(y, "b l n d -> b n l d")
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaMixedSpaceSharedBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_mix_time":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaMixedTimeBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba_fwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.mamba_bwd = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_mix_time uses official mamba-ssm and requires CUDA tensors."
                        )
                    b, n, l, _ = x.shape
                    x_mix = rearrange(self.norm(x), "b n l d -> b (n l) d")
                    y_fwd = self.mamba_fwd(x_mix)
                    y_bwd = torch.flip(self.mamba_bwd(torch.flip(x_mix, dims=[1])), dims=[1])
                    w = torch.sigmoid(self.alpha)
                    y = w * y_fwd + (1.0 - w) * y_bwd
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    y = rearrange(y, "b (n l) d -> b n l d", n=n, l=l)
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaMixedTimeBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "bimamba1_mix_time_shared":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _BiMambaMixedTimeSharedBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.alpha = nn.Parameter(torch.zeros(1))
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        raise RuntimeError(
                            "bimamba1_mix_time_shared uses official mamba-ssm and requires CUDA tensors."
                        )
                    b, n, l, _ = x.shape
                    x_mix = rearrange(self.norm(x), "b n l d -> b (n l) d")
                    y_fwd = self.mamba(x_mix)
                    y_bwd = torch.flip(self.mamba(torch.flip(x_mix, dims=[1])), dims=[1])
                    w = torch.sigmoid(self.alpha)
                    y = w * y_fwd + (1.0 - w) * y_bwd
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    y = rearrange(y, "b (n l) d -> b n l d", n=n, l=l)
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[_BiMambaMixedTimeSharedBlock(self.emb_size, drop_path_p=dp_rates[i]) for i in range(self.depth)]
            )
        elif self.temporal_module == "mamba1":
            try:
                from mamba_ssm import Mamba  # type: ignore
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            class _Mamba1MixSpaceFirstUniBlock(nn.Module):
                def __init__(self, d_model: int, drop_path_p: float):
                    super().__init__()
                    self.norm = nn.LayerNorm(d_model)
                    self.mamba = Mamba(
                        d_model=d_model,
                        d_state=self_outer.mamba_d_state,
                        d_conv=self_outer.mamba_d_conv,
                        expand=self_outer.mamba_expand,
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.drop_path = DropPath(drop_path_p)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        raise RuntimeError(
                            "mamba1 uses official mamba-ssm (selective_scan CUDA op) and requires CUDA tensors. "
                            "Run with trainer.accelerator=gpu (or switch temporal_module=axial for CPU)."
                        )
                    # x: (b, n, l, d). Space-first token order (same as bimamba1_mix_space_shared),
                    # but **unidirectional backward** scan.
                    b, n, l, _ = x.shape
                    x_mix = rearrange(x, "b n l d -> b l n d")
                    x_mix = rearrange(self.norm(x_mix), "b l n d -> b (l n) d")
                    y = torch.flip(self.mamba(torch.flip(x_mix, dims=[1])), dims=[1])
                    y = self.dropout(y)
                    y = self.drop_path(y)
                    y = rearrange(y, "b (l n) d -> b l n d", l=l, n=n)
                    y = rearrange(y, "b l n d -> b n l d")
                    return x + y

            self_outer = self
            dp_rates = [self.drop_path * i / max(self.depth - 1, 1) for i in range(self.depth)]
            self.pos_enc = SegmentPositionalEncoding(self.emb_size, self.num_segments, self.seg_length)
            self.temporal = nn.Sequential(
                *[
                    _Mamba1MixSpaceFirstUniBlock(self.emb_size, drop_path_p=dp_rates[i])
                    for i in range(self.depth)
                ]
            )
        else:
            raise ValueError(
                f"Unknown temporal_module={self.temporal_module!r}. "
                "Expected 'axial', 'mamba1', 'bimamba1', 'bimamba1_shared', 'bimamba1_st', 'bimamba1_st_shared', 'bimamba1_s', 'bimamba1_s_shared', 'bimamba1_mix_space', 'bimamba1_mix_space_shared', 'bimamba1_mix_time', or 'bimamba1_mix_time_shared'."
            )
        self.classifier = ClassificationHead(self.emb_size, output_size)

    def forward(self, features, segment_size, test_mode):
        features = self.projection(features)

        if test_mode:
            features = rearrange(
                features,
                "(b n s l) d -> b n s l d",
                n=self.num_segments,
                s=segment_size,
                l=self.seg_length,
            )
            features = rearrange(features, "b n s l d -> (b s) n l d")
        else:
            features = rearrange(
                features,
                "(b n l) d -> b n l d",
                n=self.num_segments,
                l=self.seg_length,
            )

        if self.temporal_module == "axial":
            features = rearrange(features, "b n l d -> b d n l")
            features = self.temporal(features)
            features = rearrange(features, "b d n l -> b n l d")
        elif self.temporal_module == "mamba1":
            features = self.pos_enc(features)
            features = self.temporal(features)
        elif self.temporal_module in ("bimamba1", "bimamba1_shared"):
            # Temporal scan should slide on n-axis while keeping spatial index l fixed.
            features = self.pos_enc(features)
            features = rearrange(features, "b n l d -> (b l) n d")
            features = self.temporal(features)
            features = rearrange(features, "(b l) n d -> b n l d", l=self.seg_length)
        elif self.temporal_module in ("bimamba1_st", "bimamba1_st_shared", "bimamba1_s", "bimamba1_s_shared"):
            features = self.pos_enc(features)
            features = self.temporal(features)
        elif self.temporal_module in ("bimamba1_mix_space", "bimamba1_mix_space_shared", "bimamba1_mix_time", "bimamba1_mix_time_shared"):
            features = self.pos_enc(features)
            features = self.temporal(features)

        if test_mode:
            features = rearrange(features, "(b s) n l d -> b n s l d", s=segment_size)
            features = rearrange(features, "b n s l d -> (b n s l) d")
        else:
            features = rearrange(features, "b n l d -> (b n l) d ")

        scores = self.classifier(features)

        return scores

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


class GlobalProjector(nn.Module):
    def __init__(
        self,
        in_dim=384,
        out_dim=384,
        hidden=384,
        num_downsamples=3,
        pool_hw=1,
        gn_groups=8,
        norm_type=None,
        conv_layers=None,
        **kwargs,
    ):
        super().__init__()
        self.pool_hw = int(pool_hw)

        self.mix = nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False)
        self.gn0 = nn.GroupNorm(num_groups=max(1, min(gn_groups, hidden)), num_channels=hidden)

        blocks = []
        for cfg in conv_layers:
            blocks.append(
                nn.Conv2d(
                    cfg.in_dim,
                    cfg.out_dim,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    bias=False,
                )
            )
            blocks.append(
                nn.GroupNorm(
                    num_groups=max(1, min(gn_groups, cfg.out_dim)),
                    num_channels=cfg.out_dim,
                )
            )
            blocks.append(nn.GELU())
        last_dim = conv_layers[-1].out_dim
        self.down_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))
        self.head = nn.Conv2d(last_dim, out_dim, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = F.gelu(self.gn0(self.mix(x)))  # [B, 384, 14, 14] -> [B, 384, 14, 14]
        x = self.down_blocks(x)  # [B, 384, 14, 14] -> [B, 384, 2, 2]
        x = self.pool(x)  # [B, 384, 2, 2] -> [B, 384, pool_hw, pool_hw]
        x = self.head(x)  # [B, 384, pool_hw, pool_hw] -> [B, out_dim, pool_hw, pool_hw]
        if self.pool_hw == 1:
            x = x.flatten(1) #[B, 384, 1, 1] -> [B, 384]
            return self.ln(x)
        x = x.flatten(2).transpose(1, 2).contiguous() #[B, 384, 2, 2] -> [B, 4, 384]
        return self.ln(x)


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        name,
        feature_key,
        projector="none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        # Load from the local torch hub cache when present (no network access needed).
        local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        self.base_model = (torch.hub.load(local, name, source="local") if os.path.isdir(local)
                           else torch.hub.load("facebookresearch/dinov2", name))
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.projector_name = projector  # pickled models may carry a GlobalProjector as self.projector
        self.latent_ndim = 2
        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if hasattr(self, "projector"):
            b, n, c = emb.shape
            h = w = int(n ** 0.5)
            emb = emb.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, 196, 384] -> [B, 384, 14, 14]
            emb = self.projector(emb)
            self.latent_ndim = 2 if emb.dim() == 3 else 1
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

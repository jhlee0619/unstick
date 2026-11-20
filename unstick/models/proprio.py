# Adapted from the I-JEPA code (src/models/vision_transformer.py) 
import torch.nn as nn


class ProprioceptiveEmbedding(nn.Module):
    def __init__(
        self,
        num_frames=16, # horizon
        tubelet_size=1,
        in_chans=8, # action_dim
        emb_dim=384, # output_dim
        use_3d_pos=False, # always False for now
        use_layernorm=True,
    ):
        super().__init__()

        # Map input to predictor dimension
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = nn.Conv1d(
            in_chans,
            emb_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size)
        self.norm = nn.LayerNorm(emb_dim) if use_layernorm else nn.Identity()

    def forward(self, x):
        # x: proprioceptive vectors of shape [B T D]
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x) if hasattr(self, 'norm') else x
        return x
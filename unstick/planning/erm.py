"""Residual memory: per-episode kernel regression of one-step model errors, c = sum_i w_i r_i / max(1, sum_i w_i),
w_i = exp(-|a - a_i|^2 / 2 sa^2 - d(z, z_i)^2 / 2 sz^2) (d: RMS distance over the observation dims; row b = episode b)."""
import torch


class EpisodicResidualMemory:
    def __init__(self, sigma_a, sigma_z):
        self.sigma_a, self.sigma_z = float(sigma_a), float(sigma_z)
        self.Z = self.A = self.R = None  # (B, M, P*D), (B, M, d_a), (B, M, P*D)

    def __len__(self):
        return 0 if self.Z is None else self.Z.shape[1]

    @torch.no_grad()
    def add(self, z, a, r):
        """z, r: (B, P, D) observation latents; a: (B, d_a)."""
        z, a, r = z.flatten(1)[:, None].float(), a[:, None].float(), r.flatten(1)[:, None].float()
        if self.Z is None:
            self.Z, self.A, self.R = z, a, r
        else:
            self.Z, self.A, self.R = torch.cat([self.Z, z], 1), torch.cat([self.A, a], 1), torch.cat([self.R, r], 1)

    def correction(self, z_in, a):
        """z_in: (B, P, D) input latent (observation dims), a: (B, d_a) action -> (B, P, D) correction."""
        B = z_in.shape[0]
        assert self.Z is not None and B == self.Z.shape[0], "residual memory: rollout batch must be the episode batch (GD)"
        return self.mean(z_in.flatten(1), a).view_as(z_in)

    def mean(self, zf, a):
        """zf: (B, n) flattened input latent, a: (B, d_a) -> (B, R) kernel mean of the stored values (GD rows)."""
        n = zf.shape[1]
        d2 = ((zf * zf).sum(1, keepdim=True) - 2 * torch.einsum("bn,bmn->bm", zf, self.Z)
              + (self.Z * self.Z).sum(2)).clamp_min(0) / n                          # (B, M) mean squared distance
        da2 = ((a[:, None] - self.A) ** 2).sum(2)                                   # (B, M)
        w = torch.exp(-da2 / (2 * self.sigma_a ** 2) - d2 / (2 * self.sigma_z ** 2))
        return torch.einsum("bm,bmn->bn", w, self.R) / w.sum(1, keepdim=True).clamp_min(1.0)

    def correct(self, wm, z_new, z_in, a):
        """z_new: (B, 1, P, D_full) predicted frame; z_in: (B, 1, P, D_full) its input frame; a: (B, 1, d_a)."""
        if len(self) == 0:
            return z_new
        k = wm.action_dim
        c = self.correction(z_in[:, 0, :, :-k], a[:, 0])
        return torch.cat([z_new[..., :-k] + c[:, None], z_new[..., -k:]], dim=-1)

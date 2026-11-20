import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """Latent goal cost per episode (B,): visual + alpha * proprio MSE to the goal embedding."""
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt, step=None):
        """Cost of the last predicted frame."""
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, step=None):
        """Cost of all predicted frames, weighted by base**t."""
        coeffs = np.array([base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32)
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_staged(z_obs_pred, z_obs_tgt, step=None):
        # terminal cost for the first MPC steps, then the full-horizon cost
        if step < z_obs_pred["visual"].shape[1] - 1:
            return objective_fn_last(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        else:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    elif mode == "staged":
        return objective_fn_staged
    else:
        raise NotImplementedError

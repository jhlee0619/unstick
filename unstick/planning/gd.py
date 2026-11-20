import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        lr,
        opt_steps,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            log_filename,
        )
        self.horizon = horizon
        self.action_noise = action_noise
        self.lr = lr
        self.opt_steps = opt_steps

    def init_actions(self, obs_0, actions=None):
        """Pads the warm start with zero env actions to (b, self.horizon, action_dim)."""
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
            new_actions = rearrange(
                new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
            )
            new_actions = self.preprocessor.normalize_actions(new_actions)
            new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def plan(self, obs_0, obs_g, actions=None, step=None):
        """actions: normalized warm start. Returns actions: (B, T, action_dim)."""
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)

        actions = self.init_actions(obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opt_steps, eta_min=0.0)
        n_evals = actions.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
            loss = self.objective_fn(i_z_obses, z_obs_g, step=step)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                actions += torch.randn_like(actions) * self.action_noise  # draws noise even when action_noise=0
        return actions, np.full(n_evals, np.inf)  # all actions are valid

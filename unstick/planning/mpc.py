import torch
import hydra
import numpy as np
from einops import rearrange
from utils import slice_trajdict_with_t, move_to_device
from .base_planner import BasePlanner
from .erm import EpisodicResidualMemory


class MPCPlanner(BasePlanner):
    """Replans with the sub-planner after every executed chunk, using the environment's feedback."""

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        logging_prefix="mpc",
        log_filename="logs.json",
        reuse_actions=True,
        stall_fix=None,
        stall_px=2.0,
        restart_std=1.0,
        erm=False,
        erm_sigma_a=1.0,
        erm_sigma_z=None,
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
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        self.reuse_actions = reuse_actions  # warm start each replan from the unexecuted tail of the previous plan
        # "restart": replan stalled episodes (moved < stall_px) from a random init; stall_px=1e9: at every replan
        assert stall_fix in (None, "restart"), stall_fix
        self.stall_fix, self.stall_px, self.restart_std = stall_fix, stall_px, restart_std
        # residual memory (planning/erm.py): corrects GD rollouts with this episode's observed one-step errors
        assert not erm or erm_sigma_z is not None, "erm needs erm_sigma_z"
        self.erm, self.erm_sigma_a, self.erm_sigma_z = erm, erm_sigma_a, erm_sigma_z
        self.erm_mem = None
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def _new_erm(self):
        return EpisodicResidualMemory(self.erm_sigma_a, self.erm_sigma_z)

    @torch.no_grad()
    def _erm_update(self, e_obses, action_so_far, m):
        """Stores (z_k, a_k, r_k) for the m executed model steps: r_k = observed latent of frame k+1 minus the
        uncorrected prediction from the single observed frame k, as every replan starts from one frame."""
        sp, fs = self.sub_planner, self.evaluator.frameskip
        wm, kA, K1 = sp.wm, sp.wm.action_dim, action_so_far.shape[1]
        r_rms = []
        for k in range(K1 - m, K1):
            o = move_to_device(sp.preprocessor.transform_obs(
                {"visual": e_obses["visual"][:, [k * fs, (k + 1) * fs]],
                 "proprio": e_obses["proprio"][:, [k * fs, (k + 1) * fs]]}), sp.device)  # frames k, k+1
            a = action_so_far[:, k:k + 1].to(sp.device)
            _, zp = wm.rollout({key: v[:, :1] for key, v in o.items()}, a)  # frame 1 = prediction of k+1
            zo = wm.encode(o, torch.cat([a, a], 1))  # observed latents of frames k, k+1
            r = zo[:, -1, :, :-kA] - zp[:, -1, :, :-kA]
            self.erm_mem.add(zo[:, -2, :, :-kA], a[:, -1], r)
            r_rms.append(r.pow(2).flatten(1).mean(1).sqrt())
        return torch.stack(r_rms, 1).mean(1).cpu().numpy()  # (B,) RMS residual of this chunk

    def plan(self, obs_0, obs_g, actions=None):
        """Returns actions: (B, T, action_dim) torch.Tensor (the actions argument is not used)."""
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        self.iter = 0
        self.planned_actions = []
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = None
        stalled = np.zeros(n_evals, dtype=bool)
        if self.erm:
            self.erm_mem = self._new_erm()  # empty at episode start
        while not np.all(self.is_success) and self.iter < self.max_iter:
            if self.stall_fix == "restart" and memo_actions is not None and stalled.any():
                memo_actions = memo_actions.clone()
                idx = torch.as_tensor(np.where(stalled)[0], device=memo_actions.device)
                memo_actions[idx] = torch.randn_like(memo_actions[idx]) * self.restart_std
            if self.erm:
                self.sub_planner.wm.erm = self.erm_mem  # only inside the sub-planner, not in the evaluator
            try:
                actions, _ = self.sub_planner.plan(
                    obs_0=cur_obs_0,
                    obs_g=obs_g,
                    actions=memo_actions,
                    step=self.iter,
                )  # (b, t, act_dim)
            finally:
                self.sub_planner.wm.erm = None
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :] if self.reuse_actions else None
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
            )
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            # stall: the last executed chunk moved the agent < stall_px (used by stall_fix)
            n_env = taken_actions.shape[1] * self.evaluator.frameskip
            P = np.asarray(e_obses["proprio"])
            stalled = np.linalg.norm(P[:, -1, :2] - P[:, -1 - n_env, :2], axis=-1) < self.stall_px
            if self.erm:
                logs["erm_r_rms"] = float(self._erm_update(e_obses, action_so_far, taken_actions.shape[1]).mean())
            stalled = stalled & ~self.is_success
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len

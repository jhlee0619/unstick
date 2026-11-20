import os
import gym
import json
import hydra
import random
import torch
import pickle
import warnings
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from env.venv import SubprocVectorEnv
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from planning.image_corruption import corrupt_obs_dict
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

# PointMaze-Medium (goal_source=hard): start/goal from free cells with a minimum separation, random velocities
POINT_MAZE_MEDIUM_HARD_ALL_CELLS = np.array(
    [
        (1, 1), (1, 2), (1, 5), (1, 6),
        (2, 1), (2, 2), (2, 4), (2, 5), (2, 6),
        (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 4), (4, 5), (4, 6),
        (5, 1), (5, 3), (5, 4), (5, 6),
        (6, 1), (6, 2), (6, 3), (6, 5), (6, 6),
    ],
    dtype=np.float32,
)
POINT_MAZE_MEDIUM_HARD_MIN_DIST = 3.0
POINT_MAZE_QVEL_RANGE = np.array(
    [[-5.2262554, 5.2262554], [-5.2262554, 5.2262554]], dtype=np.float32
)


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "segments":
            self.prepare_targets_from_segments(cfg_dict["eval_data_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            ood_corruption=self.cfg_dict.get("ood_corruption", None),
            ood_level=self.cfg_dict.get("ood_level", None),
        )

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        planner_cfg = self.cfg_dict["planner"].copy()
        planner_cfg["n_taken_actions"] = planner_cfg["n_taken_actions"] // self.frameskip
        planner_cfg["sub_planner"]["horizon"] = planner_cfg["sub_planner"]["horizon"] // self.frameskip
        self.planner = hydra.utils.instantiate(
            planner_cfg,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            log_filename=self.log_filename,
        )
        self.dump_targets()

    def prepare_targets(self):
        if self.goal_source == "maze_dist":
            # Diverse PointMaze: start/goal states sampled in planning_main; each sub-env has its own maze_spec
            dm = self.cfg_dict["_dm_goals"]
            init_state = dm["start_states"]  # (n_evals, 4)
            goal_state = dm["goal_states"]   # (n_evals, 4)

            obs_0, _ = self.env.prepare(self.eval_seed, init_state)
            obs_g, _ = self.env.prepare(self.eval_seed, goal_state)

            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = init_state
            self.state_g = goal_state
            return

        if self.goal_source == "hard":
            rand_init_state, rand_goal_state = self._sample_point_maze_hard_states()
            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
        else:  # dset: segments of the dataset's validation trajectories
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def _sample_point_maze_hard_states(self):
        init_states = []
        goal_states = []
        for cur_seed in self.eval_seed:
            rs = np.random.RandomState(cur_seed)
            rs.random()  # parity with umaze's start/goal flip draw
            start, goal = self._sample_point_maze_distant_cell_states(rs)
            init_states.append(start)
            goal_states.append(goal)
        return np.stack(init_states), np.stack(goal_states)

    def _sample_point_maze_distant_cell_states(self, rs):
        """Sample a start/goal pair from all hard cells with minimum distance."""
        cells = POINT_MAZE_MEDIUM_HARD_ALL_CELLS
        while True:
            i, j = rs.choice(len(cells), size=2, replace=False)
            if np.linalg.norm(cells[i] - cells[j]) >= POINT_MAZE_MEDIUM_HARD_MIN_DIST:
                break
        start = self._sample_point_maze_cell_state(rs, cells[i : i + 1])
        goal = self._sample_point_maze_cell_state(rs, cells[j : j + 1])
        return start, goal

    def _sample_point_maze_cell_state(self, rs, cells):
        loc = cells[rs.randint(len(cells))]
        qpos = loc + np.asarray(rs.uniform(low=-0.25, high=0.25, size=2), dtype=np.float32)
        qvel = np.array(
            [
                rs.uniform(low=POINT_MAZE_QVEL_RANGE[0][0], high=POINT_MAZE_QVEL_RANGE[0][1]),
                rs.uniform(low=POINT_MAZE_QVEL_RANGE[1][0], high=POINT_MAZE_QVEL_RANGE[1][1]),
            ],
            dtype=np.float32,
        )
        return np.concatenate([qpos, qvel], axis=0)

    def _apply_color_overrides(self, env_info):
        """Eval-time color overrides (env_kwargs_override.{color,agent_color,goal_color}) into env_info."""
        ekw = self.cfg_dict.get("env_kwargs_override", None) or {}
        block_color = ekw.get("color")
        agent_color = ekw.get("agent_color")
        goal_color = ekw.get("goal_color")
        if not block_color and not agent_color and not goal_color:
            return
        for ei in env_info:
            if block_color:
                ei["color"] = block_color
            if agent_color:
                ei["agent_color"] = agent_color
            if goal_color:
                ei["goal_color"] = goal_color

    def prepare_targets_from_segments(self, file_path):
        """Load pre-sampled segments (init state + actions), replay in env to get observations."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        segments = data["segments"]
        self.goal_H = data["traj_len"]

        rng = random.Random(self.cfg_dict["seed"])
        chosen = rng.sample(segments, min(self.n_evals, len(segments)))

        # Pad 5-dim states to 7-dim (add zero velocities) if needed
        raw_init = np.array([s["states"][0] for s in chosen])
        if raw_init.shape[-1] == 5:
            init_states = np.zeros((len(chosen), 7), dtype=np.float32)
            init_states[:, :5] = raw_init
        else:
            init_states = raw_init
        seg_actions = torch.stack([torch.tensor(s["actions"], dtype=torch.float32) for s in chosen])

        env_info = [{"shape": s["shape"]} if "shape" in s else {} for s in chosen]
        self._apply_color_overrides(env_info)
        self.env.update_env(env_info)

        # PushT/PushObj store actions in pixel-space.
        action_scale = 100.0 if self.env_name in ("pusht", "pushobj") else 1.0
        env_actions = seg_actions.numpy() / action_scale
        rollout_obses, rollout_states = self.env.rollout(
            self.eval_seed, init_states, env_actions
        )
        ood_corruption = self.cfg_dict.get("ood_corruption", None)
        if ood_corruption:
            base_seed = (
                self.eval_seed[0]
                if isinstance(self.eval_seed, (list, tuple, np.ndarray))
                else self.eval_seed
            )
            rollout_obses = corrupt_obs_dict(
                rollout_obses, ood_corruption, self.cfg_dict.get("ood_level", None),
                seed=int(base_seed),
            )
        self.obs_0 = {
            key: np.expand_dims(arr[:, 0], axis=1)
            for key, arr in rollout_obses.items()
        }
        self.obs_g = {
            key: np.expand_dims(arr[:, -1], axis=1)
            for key, arr in rollout_obses.items()
        }
        self.state_0 = init_states
        self.state_g = rollout_states[:, -1]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "goal_H": self.goal_H,
                },
                f,
            )

    def perform_planning(self):
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
        )
        logs, successes, e_obses, _ = self.evaluator.eval_actions(
            actions.detach(), action_len
        )
        if self.cfg_dict.get("save_exec_rollouts"):
            # executed trajectories at model stride (data for on-policy fine-tuning, scripts/finetune.py)
            fs = self.frameskip
            torch.save(
                {
                    "visual": e_obses["visual"][:, ::fs],
                    "proprio": e_obses["proprio"][:, ::fs],
                    "actions": actions.detach().cpu(),
                    "alive": np.minimum(action_len, actions.shape[1]),
                },
                "exec_rollouts.pt",
            )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    from models.dino import DinoV2Encoder
    _ = DinoV2Encoder('dinov2_vits14', 'x_norm_patchtokens')  # puts the torch.hub DINOv2 code on the path for unpickling
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")
    if not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    # checkpoints pickled with newer DINOv2 code store attn_drop as a float; wrap it (inactive in eval)
    for m in model.modules():
        if isinstance(getattr(m, "attn_drop", None), float):
            m.attn_drop = torch.nn.Dropout(m.attn_drop)
    model.to(device)
    return model


def planning_main(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath(cfg_dict["ckpt_base_path"])
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    if cfg_dict.get("goal_source") == "segments":
        # segment targets carry states and actions: a statistics-only dataset stub, no training data needed
        from types import SimpleNamespace
        transform = hydra.utils.instantiate(model_cfg.env.dataset.transform)
        from datasets.pusht_dset import (
            ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD,
        )
        dset = SimpleNamespace(
            action_dim=2,
            action_mean=ACTION_MEAN, action_std=ACTION_STD,
            state_mean=STATE_MEAN[:5], state_std=STATE_STD[:5],
            proprio_mean=PROPRIO_MEAN[:4], proprio_std=PROPRIO_STD[:4],
            transform=transform,
        )
    else:
        if cfg_dict.get("eval_data_path"):
            model_cfg.env.dataset.data_path = cfg_dict["eval_data_path"]
        _, dset = hydra.utils.call(
            model_cfg.env.dataset,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # eval-time env overrides (colors, dynamics)
    env_kwargs = dict(model_cfg.env.kwargs) if model_cfg.env.kwargs else {}
    if cfg_dict.get("env_kwargs_override"):
        env_kwargs.update(dict(cfg_dict["env_kwargs_override"]))

    if model_cfg.env.name == "diverse_maze" and cfg_dict.get("goal_source") == "maze_dist":
        # Diverse PointMaze: BFS distance-controlled (start, goal) pairs on the held-out layouts, one env per episode
        from datasets.diverse_maze_goals import generate_diverse_maze_goals

        eval_root = cfg_dict.get("eval_data_path") or str(model_cfg.env.dataset.data_path)
        dm_goals = generate_diverse_maze_goals(
            val_data_path=os.path.join(eval_root, "val"),
            n_evals=cfg_dict["n_evals"],
            seed=cfg_dict["seed"],
            min_block_radius=cfg_dict.get("min_block_radius", 4),
            max_block_radius=cfg_dict.get("max_block_radius", 10000),
        )
        cfg_dict["_dm_goals"] = dm_goals
        env = SubprocVectorEnv(
            [
                (
                    lambda spec=dm_goals["map_specs"][i], kw=env_kwargs: gym.make(
                        "diverse_maze", maze_spec=spec, **kw
                    )
                )
                for i in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
    )
    return plan_workspace.perform_planning()


@hydra.main(config_path="conf", config_name="plan_pushobj")
def main(cfg: OmegaConf):
    planning_main(cfg_to_dict(cfg))


if __name__ == "__main__":
    main()

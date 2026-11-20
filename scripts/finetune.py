"""On-policy fine-tuning (FT): fine-tune the world model offline on the base planner's own executions.

Data: <runs>/<setting>__gd__heldout__s{11,12,13}/{exec_rollouts.pt, plan_targets.pkl} from `run_eval.py --table
ft-data`; executions that start like an evaluation episode are dropped. Trains the last predictor block + final norm
(lr 5e-4) and the encoder head (lr 1e-5) with the sliding-window one-step latent MSE (3 epochs, batch 8, seed 0).

  python scripts/finetune.py MODEL SETTING [SETTING ...] --ckpt-root CKPT --data-root DATA --runs RUNS --out FT
  -> FT/MODEL/{hydra.yaml, checkpoints/model_latest.pth}
"""
import argparse, pickle, random, shutil, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE = Path(__file__).resolve().parents[1] / "unstick"
sys.path.insert(0, str(CODE))
from omegaconf import OmegaConf  # noqa: E402
from plan import load_model  # noqa: E402
from preprocessor import Preprocessor  # noqa: E402
from utils import move_to_device  # noqa: E402
from datasets.img_transforms import default_transform  # noqa: E402
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD  # noqa: E402

PRE = Preprocessor(ACTION_MEAN, ACTION_STD, STATE_MEAN[:5], STATE_STD[:5], PROPRIO_MEAN[:4], PROPRIO_STD[:4],
                   default_transform(img_size=224))  # pushing models: fixed normalization constants
HELDOUT = (11, 12, 13)
EPOCHS, BATCH = 3, 8
_SH = ("T", "L", "Z", "+", "I", "small_tee", "square")
# (target pool, seed) of every evaluation episode of the pushing models (seeds 1-10; plan.py draws 50 per seed)
EVAL = {"pushobj_shape_shift": [(s, k) for s in _SH for k in (1, 2, 3)] + [(s, k) for s in ("T", "I", "square")
                                                                          for k in range(4, 11)],
        "pusht_visual_shift": [("T", k) for k in range(1, 11)],
        "tsdino_pusht": [("T", k) for k in range(1, 11)]}
# PointMaze: start states depend on the seed; read from the base-planner evaluation runs (seeds 1-3)
EVAL_RUNS = {"mediummaze_dynamics_shift": ("maze_default", "maze_lowmass", "maze_highdamp"),
             "diversemaze_layout_shift": ("diversemaze",)}
N_TRAIN = {"pushobj_shape_shift": 379, "pusht_visual_shift": 89}  # executions kept (random subset); others: all


class Trainer:
    """Trainable tensors and one-step latent prediction loss (as in AdaJEPA test-time adaptation)."""

    def __init__(self, wm, lr, encoder_lr, finetune_encoder):
        self.wm, self.lr, self.encoder_lr, self.finetune_encoder = wm, lr, encoder_lr, finetune_encoder
        self.device = next(wm.parameters()).device
        self.criterion = nn.MSELoss()
        self._ada_predictor_params = (list(wm.predictor.transformer.layers[-1].parameters())
                                      + list(wm.predictor.transformer.norm.parameters()))
        self._ada_encoder_params = self._select_encoder_params() if finetune_encoder else []

    def _select_encoder_params(self):
        encoder = self.wm.encoder
        if hasattr(encoder, "base_model"):  # frozen DINOv2 backbone: projector / non-backbone params only
            if hasattr(encoder, "projector") and getattr(encoder, "projector_name", None) in ("channel", "global"):
                return list(encoder.projector.parameters())
            return [p for name, p in encoder.named_parameters() if not name.startswith("base_model.")]
        return list(list(encoder.named_children())[-1][1].parameters())  # last submodule

    @staticmethod
    def _set_requires_grad(module, ada_params, enabled):
        for p in module.parameters():
            p.requires_grad_(False)
        if enabled:
            for p in ada_params:
                p.requires_grad_(True)

    def _make_optimizer(self):
        groups = [{"params": self._ada_predictor_params, "lr": self.lr}]
        if self.finetune_encoder:
            groups.append({"params": self._ada_encoder_params, "lr": self.encoder_lr})
        return torch.optim.Adam(groups)

    def _prepare_segment(self, obs, act):
        # pad a dummy action for the last frame (never enters the loss)
        obs = move_to_device({k: v.clone() for k, v in obs.items()}, self.device)
        act = act.to(self.device)
        act = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
        return obs, act

    def _prediction_loss(self, z, detach_src=True, detach_tgt=True):
        # sliding-window one-step MSE on observation tokens; z: (b, T+1, p, d)
        T = z.shape[1] - 1
        if T < 1:
            return torch.tensor(0.0, device=self.device)
        window = min(self.wm.num_hist, T)
        losses = []
        for t in range(T - window + 1):
            z_src = z[:, t : t + window]
            z_tgt = z[:, t + 1 : t + 1 + window]
            if detach_src:
                z_src = z_src.detach()
            if detach_tgt:
                z_tgt = z_tgt.detach()
            z_pred = self.wm.predict(z_src)
            if self.wm.concat_dim == 0:
                loss = self.criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :])
            else:
                drop = self.wm.action_dim
                loss = self.criterion(z_pred[:, :, :, :-drop], z_tgt[:, :, :, :-drop])
            losses.append(loss)
        return torch.stack(losses).mean()


def eval_starts(a):
    """Start states of the evaluation episodes, sampled as in plan.prepare_targets_from_segments."""
    if a.model == "dinowm_pusht":  # dataset goals drawn by seed: every DINO-WM evaluation run in RUNS
        fs = [f for f in Path(a.runs).glob("dinowm_pusht__*__s*/plan_targets.pkl")
              if int(f.parent.name.rsplit("__s", 1)[1]) not in HELDOUT]
        print(f"eval starts from {len(fs)} DINO-WM runs", flush=True)
        return np.concatenate([pickle.load(open(f, "rb"))["state_0"][:, :5] for f in fs])
    if a.model in EVAL_RUNS:
        return np.concatenate([pickle.load(open(Path(a.runs) / f"{s}__gd__base__s{k}" / "plan_targets.pkl", "rb"))
                               ["state_0"][:, :5] for s in EVAL_RUNS[a.model] for k in (1, 2, 3)])
    pool = Path(a.data_root) / "pushobj_eval"
    pools = {s: pickle.load(open(pool / f"val_{s}" / "plan_targets.pkl", "rb"))["segments"] for s, _ in EVAL[a.model]}
    return np.array([c["states"][0][:5] for s, k in EVAL[a.model] for c in random.Random(k).sample(pools[s], 50)])


def load_episodes(a):
    s0_eval = eval_starts(a)
    eps, dropped = [], 0
    for s in a.settings:
        for k in HELDOUT:
            fd = Path(a.runs) / f"{s}__gd__heldout__s{k}"
            d = torch.load(fd / "exec_rollouts.pt")
            s0 = pickle.load(open(fd / "plan_targets.pkl", "rb"))["state_0"][:, :5]
            for i in range(len(s0)):
                if np.abs(s0_eval - s0[i]).max(1).min() < 1e-3:
                    dropped += 1
                    continue
                eps.append((d["visual"][i], np.asarray(d["proprio"])[i], d["actions"][i]))
    print(f"{len(eps)} executions ({dropped} dropped: same start as an evaluation episode)", flush=True)
    n = N_TRAIN.get(a.model)
    if n is None:
        return eps
    assert len(eps) >= n, (len(eps), n)
    keep = np.random.RandomState(0).choice(len(eps), n, replace=False)
    print(f"subsampled to {n}", flush=True)
    return [eps[j] for j in sorted(keep)]


def make_pre(a, cfg):
    """Pushing models: fixed constants. DINO-WM and PointMaze: dataset statistics (PointMaze: see --norm-stats)."""
    if a.model in EVAL:
        return PRE
    import hydra
    if a.model == "dinowm_pusht":
        cfg.env.dataset.data_path = str(Path(a.data_root) / "pusht_noise")
    # diversemaze: --norm-stats val (default) uses the statistics planning uses; medium: PointMaze-Medium statistics
    elif a.model == "diversemaze_layout_shift" and a.norm_stats == "val":
        cfg.env.dataset.data_path = str(Path(a.data_root) / "diverse_maze_valstats")
    else:
        cfg.env.dataset.data_path = str(Path(a.data_root) / "point_maze_medium")
    _, d = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    d = d["valid"]
    return Preprocessor(d.action_mean, d.action_std, d.state_mean, d.state_std, d.proprio_mean, d.proprio_std,
                        d.transform)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("settings", nargs="+")
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--predictor-only", action="store_true", help="do not adapt the encoder head")
    ap.add_argument("--micro-batch", type=int, default=BATCH, help="gradient accumulation (same batch of 8)")
    ap.add_argument("--norm-stats", choices=("medium", "val"), default="val",
                    help="PointMaze normalization statistics (see make_pre)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    root, out = Path(a.ckpt_root) / a.model, Path(a.out) / a.model

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    cfg = OmegaConf.load(root / "hydra.yaml")
    src = root / "checkpoints/model_latest.pth"
    pre = make_pre(a, cfg.copy())
    wm = load_model(src, cfg, cfg.num_action_repeat, torch.device(a.device))
    wm.eval()
    tr = Trainer(wm, lr=5e-4, encoder_lr=1e-5, finetune_encoder=not a.predictor_only)
    eps = load_episodes(a)
    perm = rng.permutation(len(eps))
    val, train = [eps[j] for j in perm[: len(eps) // 10]], [eps[j] for j in perm[len(eps) // 10:]]

    def batch(b):
        obs = pre.transform_obs({"visual": np.stack([e[0] for e in b]), "proprio": np.stack([e[1] for e in b])})
        return obs, torch.stack([e[2] for e in b])

    @torch.no_grad()
    def val_loss():
        return float(np.mean([tr._prediction_loss(wm.encode(*tr._prepare_segment(*batch(val[j:j + BATCH]))))
                              .item() for j in range(0, len(val), BATCH)]))

    print(f"val 1-step latent loss before {val_loss():.5f}", flush=True)
    opt = tr._make_optimizer()
    for ep in range(EPOCHS):
        tr._set_requires_grad(wm.predictor, tr._ada_predictor_params, True)
        tr._set_requires_grad(wm.encoder, tr._ada_encoder_params, True)
        wm.predictor.train(); wm.encoder.train()
        order, losses = rng.permutation(len(train)), []
        mb = a.micro_batch
        for j in range(0, len(order), BATCH):
            idx, tot = order[j:j + BATCH], 0.0
            opt.zero_grad()
            for i in range(0, len(idx), mb):
                obs, act = tr._prepare_segment(*batch([train[t] for t in idx[i:i + mb]]))
                loss = tr._prediction_loss(wm.encode(obs, act), detach_src=False,
                                           detach_tgt=bool(getattr(wm, "stop_grad", True))) * len(idx[i:i + mb]) / len(idx)
                loss.backward(); tot += loss.item()
            opt.step()
            losses.append(tot)
        wm.predictor.eval(); wm.encoder.eval()
        tr._set_requires_grad(wm.predictor, tr._ada_predictor_params, False)
        tr._set_requires_grad(wm.encoder, tr._ada_encoder_params, False)
        print(f"epoch {ep + 1}: train {np.mean(losses):.5f}  val {val_loss():.5f}  ({len(losses)} steps)", flush=True)

    payload = torch.load(src, map_location="cpu")
    payload["predictor"], payload["encoder"] = wm.predictor.cpu(), wm.encoder.cpu()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "checkpoints/model_latest.pth")
    shutil.copy(root / "hydra.yaml", out / "hydra.yaml")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()

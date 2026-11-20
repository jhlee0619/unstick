"""Evaluation runs (GD planner + MPC, evaluation seeds 1-3, 50 episodes per seed).

Each run calls unstick/plan.py once and writes <runs>/<setting>__gd__<arm>__s<seed>/logs.json (success rate after
every MPC step and at the end); a run is skipped if it is already done.

  python scripts/run_eval.py --table shape --ckpt-root CKPT --ft-root FT --data-root DATA --runs RUNS [--gpus 0,1]
  tables: shape, visual, ablation, models, pointmaze; ft-data: base-planner executions on separate episodes (seeds 11-13),
  the input of finetune.py.
  CKPT/<model>, FT/<model>: hydra.yaml and checkpoints/model_latest.pth (FT: written by finetune.py), <model> =
  pushobj_shape_shift, pusht_visual_shift, mediummaze_dynamics_shift, diversemaze_layout_shift, dinowm_pusht, tsdino_pusht
"""
import argparse, itertools, os, subprocess, sys, time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "unstick"

# setting -> (config env suffix, model, eval data (relative to --data-root), extra overrides, MPC steps)
SETTINGS = {}
for s in ["T", "L", "Z", "+", "I", "small_tee", "square"]:  # PushObj: seen shapes T L Z +, unseen I, small T, square
    SETTINGS[f"shape_{s}"] = ("pushobj", "pushobj_shape_shift", f"pushobj_eval/val_{s}/plan_targets.pkl", [], 30)
VIS = {  # PushT model evaluated on the T pool
    "clean": [], "blur": ["ood_corruption=blur"], "snp": ["ood_corruption=snp1"],
    "redagent": ["++env_kwargs_override.agent_color=Red"],
    "redblock": ["++env_kwargs_override.color=Red"],
    "redanchor": ["++env_kwargs_override.goal_color=Red"],
    "dark09": ["ood_corruption=dark", "ood_level=0.9"],
}
for k, ov in VIS.items():
    SETTINGS[f"visual_{k}"] = ("pushobj", "pusht_visual_shift", "pushobj_eval/val_T/plan_targets.pkl", ov, 30)
MAZE = {"default": [], "lowmass": ["++env_kwargs_override.density_scale=0.2"],
        "highdamp": ["++env_kwargs_override.damping_scale=20"]}
for k, ov in MAZE.items():
    SETTINGS[f"maze_{k}"] = ("maze", "mediummaze_dynamics_shift", "point_maze_medium", ov, 20)
# unseen layouts; normalization statistics from the validation split (DATA/diverse_maze_valstats, see README)
SETTINGS["diversemaze"] = ("diversemaze", "diversemaze_layout_shift", "diverse_maze_valstats", [], 20)
SETTINGS["dinowm_pusht"] = ("dinowm", "dinowm_pusht", "pusht_noise", [], 20)  # DINO-WM PushT, its GD protocol
SETTINGS["tsdino_clean"] = ("pushobj", "tsdino_pusht", "pushobj_eval/val_T/plan_targets.pkl", [], 30)  # Temporal Straightening

MEM = ["+planner.erm=true"]  # residual memory
REINIT = ["+planner.stall_fix=restart", "++planner.stall_px=1e9"]  # random init at every replan
# arm -> (hydra overrides, uses the fine-tuned model)
ARMS = {
    "base": ([], False),
    "mem": (MEM, False),
    "reinit": (REINIT, False),
    "ft": ([], True),
    "unstick_mem": (MEM, True),
    "unstick_reinit": (REINIT, True),
    "heldout": (["+save_exec_rollouts=true"], False),  # base planner on the fine-tuning episodes; saves executions
}
SH7 = tuple(f"shape_{s}" for s in ("T", "L", "Z", "+", "I", "small_tee", "square"))
VIS7 = tuple(f"visual_{v}" for v in ("clean", "blur", "snp", "dark09", "redagent", "redanchor", "redblock"))
MAZE4 = ("maze_default", "maze_lowmass", "maze_highdamp", "diversemaze")
MAIN = ["base", "unstick_mem", "unstick_reinit"]
# table -> [(settings, arms, seeds)]
TABLES = {
    "shape": [(SH7, MAIN, (1, 2, 3))],
    "visual": [(VIS7, MAIN, (1, 2, 3))],
    "ablation": [(("shape_T", "shape_I", "shape_square", "visual_clean", "visual_redblock"),
                  ["base", "mem", "reinit", "ft", "unstick_mem", "unstick_reinit"], (1, 2, 3))],
    "models": [(("dinowm_pusht", "tsdino_clean"), MAIN, (1, 2, 3))],
    "pointmaze": [(MAZE4, ["base", "mem", "reinit", "ft", "unstick_mem", "unstick_reinit"], (1, 2, 3))],
    "ft-data": [(("shape_T", "shape_L", "shape_Z", "shape_+", "visual_clean", "maze_default", "diversemaze",
                  "dinowm_pusht", "tsdino_clean"), ["heldout"], (11, 12, 13))],
}
# residual-memory kernel width sigma_z per model (fine-tuned models use the base value)
SIGMA_Z = {"pushobj_shape_shift": 0.01742, "pusht_visual_shift": 0.07861, "mediummaze_dynamics_shift": 0.33710,
           "dinowm_pusht": 0.34987, "diversemaze_layout_shift": 0.0449, "tsdino_pusht": 0.28365}


def command(setting, arm, seed, a, py=sys.executable):
    env, model, data, ov, iters = SETTINGS[setting]
    overrides, ft = ARMS[arm]
    run = f"{setting}__gd__{arm}__s{seed}"
    cmd = [py, "plan.py", f"--config-name=plan_{env}.yaml",
           f"ckpt_base_path={Path(a.ft_root if ft else a.ckpt_root) / model}", f"eval_data_path={Path(a.data_root) / data}",
           f"seed={seed}", f"planner.max_iter={iters}", f"hydra.run.dir={Path(a.runs) / run}", *ov, *overrides]
    if "+planner.erm=true" in overrides:
        cmd.append(f"+planner.erm_sigma_z={SIGMA_Z[model]}")
    return run, cmd


def jobs(table, a):
    for settings, arms, seeds in TABLES[table]:
        for s, arm, seed in itertools.product(settings, arms, seeds):
            yield command(s, arm, seed, a)


def done(runs, run):
    f = Path(runs) / run / "logs.json"
    return f.exists() and "final_eval/success_rate" in f.read_text()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, choices=list(TABLES))
    ap.add_argument("--ckpt-root", required=True, help="released checkpoints, one folder per model")
    ap.add_argument("--ft-root", default="", help="fine-tuned checkpoints from scripts/finetune.py")
    ap.add_argument("--data-root", required=True, help="released evaluation data")
    ap.add_argument("--runs", default="runs", help="output folder")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent runs per GPU")
    ap.add_argument("--dry", action="store_true", help="print the commands only")
    a = ap.parse_args()
    a.runs = str(Path(a.runs).resolve())
    todo = [(r, c) for r, c in jobs(a.table, a) if not done(a.runs, r)]
    print(f"{len(todo)} runs to do", flush=True)
    if a.dry:
        for r, c in todo:
            print(r, " ".join(c[2:]))
        return
    env = dict(os.environ)
    mj = os.path.expanduser("~/.mujoco/mujoco210/bin")  # MuJoCo 2.1 binaries for mujoco-py (PointMaze)
    env["LD_LIBRARY_PATH"] = f"{mj}:{sys.prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env.update(DATASET_DIR=str(a.data_root), PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MUJOCO_PY_FORCE_CPU="1")
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    running = {}  # run -> (process, gpu)
    while todo or running:
        for run in [r for r, (p, _) in running.items() if p.poll() is not None]:
            slots.append(running.pop(run)[1])
            print(f"[{time.strftime('%H:%M')}] {'OK ' if done(a.runs, run) else 'FAIL'} {run}", flush=True)
        while todo and slots:
            run, cmd = todo.pop(0)
            g = slots.pop(0)
            (Path(a.runs) / run).mkdir(parents=True, exist_ok=True)
            log = open(Path(a.runs) / run / "stdout.log", "w")
            running[run] = (subprocess.Popen(cmd, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                             env={**env, "CUDA_VISIBLE_DEVICES": g}), g)
        time.sleep(5)


if __name__ == "__main__":
    main()

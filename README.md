# Stuck in Place: Stall Diagnosis and Remedies for JEPA Planners

Junhyeok Lee, Dongheon Lee, Kyu Sung Choi<br>
Seoul National University

[Paper](https://arxiv.org/abs/2611.25415) · [Project page](https://jhlee0619.github.io/unstick/) · [Models](https://huggingface.co/jhlee0619/unstick)

Unstick breaks the planning-loop fixed point behind most failures of JEPA planners on pushing tasks, with fixed
weights: on-policy fine-tuning before deployment plus a residual memory of the episode's prediction errors
(Unstick-Mem) or random re-initialization at every replan (Unstick-ReInit).

## Setup

```
conda env create -f environment.yml && conda activate unstick-eval
pip install --no-deps d4rl==1.1    # PointMaze; also needs the MuJoCo 2.1 binaries in ~/.mujoco/mujoco210
```
Data and models (each model in `CKPT/<model>/{hydra.yaml, checkpoints/model_latest.pth}`, the data in `DATA/`):
* [AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa): models `pushobj_shape_shift`, `pusht_visual_shift`, `mediummaze_dynamics_shift`, `diversemaze_layout_shift`; data `pushobj_eval/`, `point_maze_medium/`, `diverse_maze/`.
* [DINO-WM](https://github.com/gaoyuezhou/dino_wm): PushT model `dinowm_pusht` and data `pusht_noise/`.
* [Temporal Straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening): PushT model `tsdino_pusht` (frozen DINOv2 encoder).
* [DINOv2](https://github.com/facebookresearch/dinov2): ViT-S/14, loaded through `torch.hub`.
* [D4RL](https://github.com/Farama-Foundation/D4RL): PointMaze environment (MuJoCo).

Also create `DATA/diverse_maze_valstats/{train,val}`, both linking to `diverse_maze/val` (normalization statistics).

## Fine-tuning and evaluation

Fine-tuned models: `hf download jhlee0619/unstick --local-dir FT` gives `FT/<model>/{hydra.yaml, checkpoints/model_latest.pth, NOTICE}`
for `--ft-root FT` ([Hugging Face](https://huggingface.co/jhlee0619/unstick), MIT; `diversemaze_layout_shift` uses `--norm-stats val`). To fine-tune yourself:
```
A="--ckpt-root CKPT --data-root DATA --runs RUNS"
python scripts/run_eval.py --table ft-data $A          # base-planner executions on separate episodes
python scripts/finetune.py pushobj_shape_shift shape_T shape_L shape_Z shape_+ $A --out FT
python scripts/finetune.py pusht_visual_shift visual_clean $A --out FT
python scripts/finetune.py mediummaze_dynamics_shift maze_default $A --out FT
python scripts/finetune.py diversemaze_layout_shift diversemaze $A --out FT
python scripts/finetune.py dinowm_pusht dinowm_pusht $A --out FT
python scripts/finetune.py tsdino_pusht tsdino_clean --predictor-only $A --out FT
python scripts/run_eval.py --table shape $A --ft-root FT    # tables: shape, visual, ablation, models, pointmaze
```
Each run writes `RUNS/<setting>__gd__<arm>__s<seed>/logs.json` (`final_eval/success_rate`); `--dry` prints the commands.

## License and citation

MIT (`LICENSE`). `unstick/` is a modified copy of the AdaJEPA code (`LICENSES/MIT-AdaJEPA.txt`); see `THIRD_PARTY.md`.

```bibtex
@article{lee2026stuck,
  title   = {Stuck in Place: Stall Diagnosis and Remedies for {JEPA} Planners},
  author  = {Lee, Junhyeok and Lee, Dongheon and Choi, Kyu Sung},
  journal = {arXiv preprint arXiv:2611.25415},
  year    = {2026}
}
```

# Third-party code

`unstick/` is a modified copy of the released AdaJEPA code (commit 51d8665b7978); its license, with the copyright
notice as shipped, is in `LICENSES/MIT-AdaJEPA.txt`. Our changes add the residual memory (`planning/erm.py`) and
random re-initialization (`planning/mpc.py`), the evaluation configs (`conf/`) and checkpoint-loading fixes
(`plan.py`), and remove unused code. The trainable-parameter selection and loss in `scripts/finetune.py` are adapted
from the AdaJEPA code (`planning/adajepa.py`, same license).

Files that carry code from other projects (source notes are kept in the files):

| File | Source | License |
|---|---|---|
| `unstick/models/vqvae.py` | DeepMind Sonnet VQ-VAE, ported to PyTorch | Apache License 2.0 (notice in the file; text in `LICENSES/Apache-2.0.txt`) |
| `unstick/models/proprio.py` | I-JEPA (`src/models/vision_transformer.py`) | as distributed with the AdaJEPA code |
| `unstick/models/vit.py` | vit-pytorch (`vit_pytorch/vit.py`) | as distributed with the AdaJEPA code |
| `unstick/env/venv.py` | Tianshou (`SubprocVectorEnv`) | as distributed with the AdaJEPA code |
| `unstick/env/pusht/pusht_env.py` | Implicit Behavior Cloning (PushT) | as distributed with the AdaJEPA code |
| `unstick/env/pointmaze/` | D4RL 1.1 point-maze environment | Apache License 2.0 (D4RL; text in `LICENSES/Apache-2.0.txt`) |

`LICENSES/Apache-2.0.txt` is the Apache License 2.0 text; its two web-address lines are omitted (title line; the
appendix example now points to the file itself).

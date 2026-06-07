# nvidia-nemotron-reasoning

Supervised Fine-Tuning (SFT) of NVIDIA Nemotron 3 Nano 30B for the NVIDIA Nemotron Model Reasoning Challenge.

## Result

**Public leaderboard: 0.61** (best submission, adapter v2).
Featured competition — ~4,041 teams — deadline 2026-06-15.
Competition page: [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

## Model and approach

| Item | Detail |
|---|---|
| Base model | `metric/nemotron-3-nano-30b-a3b-bf16` (Nemotron 3 Nano 30B-A3B, BF16, ~3B active parameters via MoE) |
| Fine-tuning method | Supervised Fine-Tuning with Low-Rank Adaptation (LoRA) |
| Training signal | Chain-of-Thought (CoT) reasoning traces |
| LoRA config | rank 32, alpha 64, dropout 0.05, targets `in_proj / out_proj / up_proj / down_proj` |
| Precision | BF16 throughout |
| Optimizer | `adamw_torch_fused`, cosine LR schedule, gradient checkpointing |

Training ran entirely on Kaggle GPU (NVIDIA RTX Pro 6000) with `enable_internet=false` (offline kernel). The Kaggle BYOD (Bring Your Own Docker) image was used to satisfy the competition runtime requirements.

## Submissions

| Adapter | Public LB | Status |
|---|---|---|
| v2 (nemotron-sft-adapter-v1) | **0.61** | COMPLETE — best result |
| v4 | 0.58 | COMPLETE |

## Notebooks

| Notebook | Role |
|---|---|
| `nemotron-sft-cot-training.ipynb` | Builds the CoT training pipeline from scratch: loads the base model, applies LoRA, loads the custom CoT dataset from `nemotron-cot-training-data`, trains for 3 epochs (seq len 2048, effective batch 8), and saves the LoRA adapter + `submission.zip`. First working version with verbose path resolution and dataset statistics. |
| `nemotron-sft-v2.ipynb` | Refactored iteration of the training pipeline: adds the `ptxas-blackwell` binary patch required by the Kaggle BYOD environment, condenses the setup boilerplate, and reproduces the same 3-epoch training run. This adapter achieved public LB **0.61**. |
| `nemotron-sft-v3.ipynb` | Third iteration: same architecture as v2 but switches `eval_strategy` from `steps` to `no` and moves to epoch-level checkpointing (`save_strategy='epoch'`, `save_total_limit=1`), removing mid-training evaluation overhead. |
| `nemotron-sft-v4.ipynb` | Sequence-length experiment: reduces `MAX_SEQ_LENGTH` from 2048 to 1024 based on observed token-length distribution of the CoT data (mean ~419 tokens, max ~657). Doubles per-device batch size to 2 and halves gradient accumulation to 4, keeping effective batch size constant. Public LB 0.58. |
| `nemotron-sft-v5.ipynb` | Single-epoch experiment with the v3 CoT dataset variant (includes `<think>` tags and weighted oversampling of under-represented reasoning types): trains for 1 epoch instead of 3 to reduce overfitting on the CoT template format. Disables checkpointing (`save_strategy='no'`). |
| `nemotron-submit.ipynb` | Submission-only utility: locates a pre-trained LoRA adapter from a Kaggle dataset input, copies `adapter_config.json` and `adapter_model.safetensors` to the working directory, and packages them into `submission.zip` for upload — no training involved. |

## Data and artifacts

The project relies on two external Kaggle assets that are **not vendored in this repository** due to size:

- **`nemotron-cot-training-data`** — custom Chain-of-Thought dataset (JSONL) used as the fine-tuning signal. Uploaded as a private Kaggle dataset and mounted at `/kaggle/input/nemotron-cot-training-data/` inside each training kernel.
- **LoRA adapters** (`nemotron-sft-adapter-v1`, `nemotron-sft-adapter-v2`) — trained adapter weights (~3.2 GB each, `adapter_model.safetensors` + `adapter_config.json`). Stored as Kaggle datasets and referenced by `nemotron-submit.ipynb` via the `/kaggle/input/` mount.

All notebooks reference these assets as Kaggle dataset sources; they resolve correctly only inside the Kaggle kernel environment.

## Hardware and runtime

- GPU: NVIDIA RTX Pro 6000 (Kaggle-hosted)
- Internet: disabled (`enable_internet=false`) — all model weights and dependencies fetched via `kagglehub` or pre-installed in the BYOD image
- Environment note: notebooks include a `ptxas-blackwell` binary patch required to work around a permission issue in the Kaggle BYOD Triton backend

## Repository structure

```
nvidia-nemotron-reasoning/
└── notebooks/
    ├── nemotron-sft-cot-training.ipynb   # CoT training pipeline v1
    ├── nemotron-sft-v2.ipynb             # Training iteration v2 (best LB)
    ├── nemotron-sft-v3.ipynb             # Training iteration v3
    ├── nemotron-sft-v4.ipynb             # Seq-len reduction experiment (v4)
    ├── nemotron-sft-v5.ipynb             # Single-epoch CoT v3 experiment
    └── nemotron-submit.ipynb             # Adapter packaging / submission utility
```

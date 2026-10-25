# Slurm cluster entrypoints

Set `PROJECT_ROOT` to your local project root, or leave it unset to use the parent of the repository directory. Shared dependency environment: `envs/ttrv` (Python 3.10, torch 2.5.1/CUDA 12.4, vLLM 0.7.3, Transformers 4.52.4). Each entry uses only its own repository's `verl` via PYTHONPATH. Other project environments are separate. Machine-specific paths and scheduler settings belong in your local environment, not in this repository.

Model paths: `ckpt/OpenGVLab/InternVL3-2B` for main/baseline; `ckpt/Qwen/Qwen2.5-VL-7B-Instruct` for Qwen. Data: `benchmark/ttrv/verl_data/<task>`. Source tables and base weights are read-only inputs to these scripts. All output/cache/temp paths are below the root.

CPU checks from the repository directory:

```bash
export PROJECT_ROOT=/path/to/project
bash cluster/run.sh cpu
bash cluster/run.sh config
```

`cpu`: dependency and trainer imports, offline processor/config/tokenizer load, decode every referenced image and read all 22 Parquet tables. `config`: same selected-task preflight then Hydra resolves the actual training configuration without launching Ray/training. `gpu-smoke`: seeded FlashAttention forward/backward against PyTorch SDPA, full checkpoint load, bounded language-head loss/backward. This is not multimodal rollout or RL validation. `train-smoke`: original trainer, one update, small copied train/test slice, two votes/samples, 32 response tokens, no checkpoint write. Original datasets and base model files are unchanged. Successful kernel/import tests alone do not certify the complete trainer; inspect separate reports.

## Existing allocation: direct SSH and node-local tmux

The current user workflow does not need sbatch or srun. SSH to the node of a current RUNNING allocation, then run in tmux:

```bash
cd "$PROJECT_ROOT/TTRV"
tmux new -s prove-check
: "${CURRENT_JOB:?Set your current allocation ID}"
"$PROJECT_ROOT/envs/ttrv/bin/python" cluster/ssh_run.py --job "$CURRENT_JOB" --mode kernel
```

Set `CURRENT_JOB` to your current allocation, and use the repository `TTRV-intuitor-baseline` or `TTRV-qwen25vl7b` as needed. This helper checks the current allocation owner, node, GPU indices and CPU set, uses the first requested allocated GPU indices only if they are idle and records their UUIDs, pins CPUs within the allocation, and checks the allocation remains RUNNING. It creates no Slurm job or step. `nvidia-smi` can list other people's GPUs; never select a card outside the verified allocation. Existing keeper processes must be handled explicitly by their owner; this helper does not terminate them. `guard.py` limits test wall time and host memory to the recorded allocation, monitors descendants and cleans up only its own processes.

`--mode kernel` tests CUDA, FlashAttention forward/backward, vLLM's CUDA RMSNorm and a small synthetic optimizer step. It does not load/train a complete checkpoint. `--mode gpu-smoke` attempts the heavier full-checkpoint language backward check; `--mode train-smoke` attempts one real trainer step; `--mode train` is formal training. The latter modes require enough authorized host memory as well as VRAM. All use the existing allocation and no sbatch/srun.

The alternate `submit.sh` supports scheduler-managed starts for future use. Setting `EXISTING_JOB=<id>` uses srun in that existing allocation; omitting it submits a new batch allocation. Before a new submission, set `SLURM_ACCOUNT`, `SLURM_QOS`, and `PARTITION` to values authorized by your cluster. Resource defaults are one GPU, four CPUs, 4G requested host memory, and 30 minutes; change `GPUS`, `CPUS`, `MEMORY`, and `WALLTIME` only within your quota. The historical setup was tested on H100; A100 is a compatibility candidate and RTX 5090 is not validated with this old CUDA stack.

Formal training is opt-in and was not launched by environment setup:

```bash
TASK=mme_20 WALLTIME=02:00:00 bash cluster/submit.sh train trainer.total_training_steps=20
```

No automatic full benchmark loop or checkpoint resume. Main/Qwen retain the low-level `verl/examples/ttrv/run.sh` reward defaults; baseline explicitly sets `REWARD_SIGNAL=intuitor` and its original self-certainty settings. For a specific PROVE experiment, pass the original method's environment/hyperparameters explicitly; this migration does not silently choose the “best” historical method. Per-task parser defaults cover the migrated 11 datasets. Additional Hydra overrides are forwarded unchanged after the defaults. `train` inherits the original low-level save/test defaults; override them explicitly for the intended run.

Logs/reports: `experiments/ttrv_cluster/<repo>/<mode>_<timestamp>_<job>_<pid>/`; Slurm batch output: `logs/ttrv_slurm/`. Every launch records its exit code; `preflight.json` is not the training completion result. Batch/step termination cleans up that step's descendants; there is no global ray stop/pkill. Check training logs, one-step metrics and Slurm exit state before declaring an experiment successful. `PROJECT_ROOT` and `CLUSTER_VERIFIED_JOB_ID` are the generic environment names used by these entrypoints.


## Verification status (2026-10-25)

- All three repos: documented CPU and config commands passed (6/6), each reading 22 Parquet tables / 22,303 rows and fully decoding 11,579 referenced images. Offline model configs/tokenizers/processors and trainer imports passed.
- Shared environment: direct SSH + node-local tmux kernel witness passed on one allocated H100 NVL (CUDA 12.4, FlashAttention forward/backward, vLLM CUDA RMSNorm, small optimizer update). Peak host PSS approximately 1.26 GiB.
- Full-checkpoint `gpu-smoke` did not complete: the explicit test guard stopped its process tree at approximately 4.03 GiB PSS (a later instrumented attempt reached 4.73 GiB before the polling guard reacted) against the existing allocation's 4 GiB requested host RAM. This was our guard, not a GPU failure or a cluster OOM kill. The instrumented report confirms the stop occurred during full checkpoint loading. Full-checkpoint backward, full multimodal rollout and one-step RL training remain unvalidated. See [MEMORY_BUDGET.md](MEMORY_BUDGET.md) for the measured stages and the current CPU-FP32 initialization / reference-offload requirements. 8G/16G/32G test-only memory requests were rejected by the current QOS; no job was submitted by those checks. Direct SSH does not increase the allocated memory entitlement.
- `installed-freeze.txt` records the resolved environment. It is an inventory, not a blindly installable requirements file: the shared runtime has a legacy dependency metadata conflict. `latex2sympy2==1.9.1` declares antlr runtime 4.7.2, whereas Hydra needs the installed 4.9.3. Simple LaTeX parsing and actual Hydra composition passed; this exception remains visible in pip check and is not a blanket mathematical-parser compatibility guarantee. Keep this combination explicit when reproducing the environment; do not silently downgrade Hydra's runtime.
- Base models, original datasets, and other project environments were not modified. This is environment and launcher preparation, not benchmark reproduction.

# Host RAM budget: measured and theoretical boundaries

This concerns host RAM, not GPU VRAM. The existing job used during setup requests 4096 MiB host RAM for the node, not 4096 MiB per GPU. The test guard is an explicit userspace PSS limit, not evidence of a cluster OOM kill. No swap, quotas, or kernel enforcement were changed.

## Model-size lower bounds

Read directly from the downloaded safetensors headers (no tensor materialization):

| Model | Stored BF16 tensor elements | BF16 storage | FP32 equivalent |
|---|---:|---:|---:|
| InternVL3-2B | 2,088,957,440 | 3.891 GiB | 7.782 GiB |
| Qwen2.5-VL-7B-Instruct | 8,292,166,656 | 15.445 GiB | 30.891 GiB |

For one complete CPU copy, `bytes = stored_elements × bytes_per_element`; divide by `2^30` for GiB. These are weight-storage lower bounds, excluding optimizer/gradient states, Python/Ray, data, loading buffers, and other process memory. Stored tensor elements are not a new claim about trainable parameter counts.

All three current repositories use CPU initialization for mesh rank 0 in `verl/verl/utils/fsdp_utils.py:get_init_weight_context_manager`. Their actor defaults to FP32 in `verl/verl/workers/fsdp_workers.py:_build_model_optimizer`; this model exists before FSDP wraps/moves/shards it. Reference FSDP additionally uses `CPUOffload(offload_params=True)` unconditionally when role is not actor. The low-level shell also enables reference offload. Thus the existing full training implementation cannot fit in a genuine 4 GiB aggregate host-RAM ceiling merely because ample GPU VRAM is available. Adding GPUs does not remove the rank-0 full CPU initialization phase.

## Measurements

- Shared CUDA/FlashAttention/vLLM kernel witness plus a tiny optimizer update: passed on an allocated H100 NVL, about 1.24–1.26 GiB peak process-tree PSS.
- Staged imports with a verified numeric GPU binding: about 761 MiB after trainer import, 804 MiB after FSDP worker import, 905 MiB after model config/tokenizer/processor. No model weights or training were involved.
- The full-checkpoint BF16 validation passed data and FlashAttention checks, then the added stage report showed `loading_full_checkpoint` when the 4 GiB guard stopped it. A 0.25-second polling guard can overshoot its limit between samples; the instrumented attempt observed about 4.73 GiB peak PSS before termination. The complete model was not reported loaded; neither its forward/backward nor RL training completed.
- Earlier attribution to dependency imports was premature: buffered stdout hid progress. The persisted stage report and subsequent explicit stage markers establish the later failure phase.
- The first UUID-bound import probe revealed that vLLM 0.7.3's platform code parses CUDA_VISIBLE_DEVICES entries as integers. Direct SSH now binds the verified allocated numeric indices and records expected UUIDs separately; the kernel witness checks the visible UUID.

## What could fit in 4 GiB?

Four GiB is not a mathematical impossibility for every GPU workload. A redesigned path could keep parameters/optimizer/reference state on GPU, load bounded chunks without a complete CPU copy, and minimize Python workers, Ray object storage and data-loader buffering. Whether this complete RL stack can fit is unproven. Changing placement must preserve the intended FP32 optimizer/master-state behavior; silently switching the optimizer to BF16 is not a validated memory fix.

Possible next work: a bounded GPU-first/streaming loader, configurable reference offload with GPU residency, and real one-step RL tests with process-tree memory measurements. Alternatively obtain an authorized larger host-RAM allocation. The current QOS rejected 8G/16G/32G `--test-only` requests; direct SSH and removing the test guard do not change the allocated memory entitlement. Setup did not implement the loading/offload redesign, remove the guard, or claim successful full training.

# R5 Reproduction Steps — Team RAMEN (Team 11)

## Overview

| Field | Value |
|------|-----|
| Model | π0.5 fine-tuned (Run73: `r71s50-pf-noeval-32d` (Run71 s50000 から継続学習), 32D output, no aux-head) |
| Submission step | s020000 (final checkpoint) |
| **Quantization** | **bf16 quantized** (deploy-only, VRAM-constrained) |
| Base model | `ICRA-2026-RAMEN/pi05-baseline-100k-pt` (organizer-published) |
| Training data | `ICRA-2026-RAMEN/airoa-public-filter` |
| **Checkpoint (R2)** | `s3://airoa-icra-team-11/r5-pi05-run73-r71s50-pf-noeval-32d-s020000/` (**bf16 quantized, 8.7 GiB**) |
| **HF Hub (deploy, bf16)** | **`ICRA-2026-RAMEN/pi05-round5-run73-r71s50-pf-noeval-32d-bf16`** |
| HF Hub (training fp32, ref) | `ICRA-2026-RAMEN/pi05-round5-run73-r71s50-pf-noeval-32d` |
| Fork repository | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| Branch | `feat/lerobot-pi05-r5-run73` |
| Backend | `lerobot` (pre-set in `.env`, no manual export needed) |
| Mode | `e2e` (pre-set in `.env`, HVLA not used, single-model approach) |
| **VRAM** | **~9.9 GB** (bf16 quantized; A100 measured. RTX 5070 Ti 16 GB has headroom) |
| **SSD** | Docker ~6.84 GB + checkpoint **~8.7 GiB** = **~14.5 GB** (within 30 GB limit) |
| Tokenizer | Bundled in container (`/workspace/tokenizer/paligemma-3b-pt-224`, no HF_TOKEN needed) |

## Major Changes in R5 (Diff from R4)

| Item | R4 | R5 |
|------|-----|-----|
| Submitted model | run52 s040000 (8D output → 11D pad, fp32 ~8.8 GB) | **Run73 s020000 bf16** (32D output → 11D extract, 8.7 GiB) |
| Training data | `airoa-sft-v5` | **`airoa-public-filter`** (public-task-focused) |
| `transformers` | 5.3.0 (nested SigLIPVisionModel) | **5.7.0** (flat SigLIPVisionModel) |
| `lerobot` fork | @ramen `c343490c` (vision_tower bug) | **@ramen `7431fb1d`** (PR #9 vision_tower fix) |
| `PI05Policy.from_pretrained` | default `strict=False` | **`strict=True`** (silent-fallback prevented) |
| HSR state pad | (8D ckpt, no pad needed) | **8D → 32D pad implementation** (`_pad_state_8d_to_32d`) |
| `config.json` `dtype` | `"float32"` | **`"bfloat16"`** (required for bf16 alloc) |
| Action postprocessor | basic EMA (α=0.3) + clip [-1.0, 1.23] | **Issue #197 fix suite** (see table below) |
| Client EMA | `action_smoothing=ema, ema_alpha=0.2` (double-EMA) | **`action_smoothing=none`** (server-only EMA, 5x faster response) |
| **VRAM** | ~9 GB | **~9.9 GB** (bf16: fp32 16.5 GB → 9.9 GB) |
| **CPU RAM peak** | ~30 GB (PI05Policy.from_pretrained double-buffering) | **~9 GB** (PR #12 new path `LEROBOT_LOW_CPU_MEM=1` default, -78%) |
| **Startup time** | ~111 s | **~6 s** (PR #12 new path, -95%) |

### Action postprocessor settings introduced by Issue #197 (PR #9)

`postprocessor` block in `hierarchical_config_optimized.yaml` (applied for both e2e and HVLA):

| Component | Value | Purpose |
|---|---|---|
| `gripper_clip` | `[-1.0, 1.239]` | HSR mechanical limit (GT q99=1.2392); fixes 33 frames clipped in R4 |
| `temporal_ensemble` | window=10, decay=0.5 | ACT-style chunk overlap (Zhao et al. 2023; matches chunk_size=10) |
| `action_smoothing` | EMA α=0.5, gripper exclude | Server-only EMA, response 5τ=0.36s (5x improvement vs old 0.3+client 0.2) |
| `head_zero_mask` | dims [6, 7] | Training GT 97%+ has \|val\|<0.001 (head stationary); prevents unintended head motion |
| `action_clip` | `dim 10 (base_theta) ±0.32` | Physical safety guard; training data has outliers up to 1.034 rad/step (593 deg/s) |
| `prompt_validation` | true | Logs prompt format for debugging (no behavior change) |

## Prerequisites

- NVIDIA GPU (**16 GB+ VRAM**; uses ~9.9 GB after bf16 quantization)
- Host RAM: **12 GB+** (new path `LEROBOT_LOW_CPU_MEM=1` default has peak ~9 GB)
  - Old path (`LEROBOT_LOW_CPU_MEM=0`) requires CPU peak ~40 GB; OOM on 24 GB hosts
- Docker (>= 20.10) + Docker Compose v2 + NVIDIA Container Toolkit
- AWS CLI (`pip install awscli`)
- Host SSD free space: 20 GB+ (Docker image ~7 GB + ckpt ~8.7 GiB + working space)
- HF_TOKEN is **not required** (PaliGemma tokenizer bundled in container)
- Internet access: build-time only; inference works fully offline (`--network none` verified)

## Reproduction Steps

### 1. Clone the repository

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05-r5-run73
```

### 2. Download the checkpoint

```bash
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 sync \
    s3://airoa-icra-team-11/r5-pi05-run73-r71s50-pf-noeval-32d-s020000/ checkpoints/r5/
```

Expected file listing (bf16 quantized):
```
checkpoints/r5/
├── config.json                                                 (2.9 KB, dtype="bfloat16")
├── model.safetensors                                           (~8.7 GiB, bf16 quantized)
├── policy_preprocessor.json                                    (2.3 KB)
├── policy_preprocessor_step_2_normalizer_processor.safetensors (3.8 KB)
├── policy_postprocessor.json                                   (663 B)
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors (3.8 KB)
└── train_config.json                                           (8.0 KB)
```

### 3. Start the container

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r5
./RUN-DOCKER-CONTAINER.sh up
```

Other environment variables (`POLICY_BACKEND=lerobot`, `POLICY_MODE=e2e`, `POLICY_SERVER_PORT=8000`) are pre-configured in `.env`; manual export is unnecessary.

### 4. Verify the policy server (sanity check)

**Important**: R4 evaluation failed (0/3) due to a `PI05Policy.from_pretrained` silent-fallback bug. Always verify the following in R5.

```bash
# Wait for startup completion (up to 300 sec), then:
docker logs airoa_policy_server 2>&1 | tail -20
```

Expected output (**all must appear**):
```
[entrypoint] Checkpoint copied to /workspace/_checkpoint
[entrypoint] config.json: compile_model True -> False
[entrypoint] config.json: gradient_checkpointing True -> False
[entrypoint] config.json: removed DAFD fields: ['use_dafd', 'dafd_gripper_dim', ...]
[entrypoint] policy_preprocessor.json: tokenizer_name google/paligemma-3b-pt-224 -> /workspace/tokenizer/paligemma-3b-pt-224
INFO:lerobot_hsr_policy:Loading PI05Policy from /workspace/_checkpoint on cuda
INFO:lerobot_hsr_policy:expected_state_dim=32 (HSR client sends 8D)         ← ★ 32D state pad active
INFO:lerobot_hsr_policy:PI05Policy loaded successfully                       ← ★ load OK (strict=True)
INFO:websockets.server:server listening on 0.0.0.0:8000                      ← ★ startup complete
```

**If any of the following appear, stop immediately and investigate**:
- `Warning: Could not load state dict` → vision_tower load failed. Check `uv.lock` for `lerobot @ ramen 7431fb1d` or later
- `RuntimeError: ... Missing key(s) in state_dict` → silent-fallback prevention is working. Confirm transformers is 5.7.0
- `RuntimeError: tensor a (8) ... b (32)` → state pad is not active. Confirm you are on `feat/lerobot-pi05-r5-run73` branch (`git log -1`)
- `RuntimeError: CUDA out of memory` → bf16 not active. Verify `cat checkpoints/r5/config.json | grep dtype` shows `"bfloat16"`
- Host process OOM-killed (e.g. 24 GB CPU RAM environment) → new path (`LEROBOT_LOW_CPU_MEM=1` default) likely disabled. Check that `.env` does not set `=0` explicitly
- `Policy config not found: /workspace/hierarchical_config.yaml (using defaults)` → ActionPostprocessor wrap is **disabled** (all Issue #197 safety guards inactive). Confirm `docker-compose.yml` bind-mount is effective and `hierarchical_config_optimized.yaml` exists at the repository root (shipped in repo)

### 5. Enter the HSR client container

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 6. Launch the HSR policy client (inside container)

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

Key launch parameters (defaults in `deploy/hsr_policy_client/launch/hsr_policy_client.launch`):

| Argument | Default | Notes |
|------|------|------|
| `gripper_mode` | `hybrid` | close=force control (effort=-0.018), open=continuous position (same as R4) |
| `prefetch_threshold` | `0` | RTC disabled, sync mode (same as R4) |
| `update_freq` | `10` | Server inference rate (Hz) |
| `adopted_action_chunks` | `10` | Number of steps adopted per chunk (matches model chunk_size=10) |
| `upsample` / `upsample_hz` / `upsample_method` | `true` / `100` / `spline` | 10Hz → 100Hz spline interpolation |
| **`action_smoothing`** | **`none`** | **Changed from `ema` in PR #9**. Server-side EMA α=0.5 handles smoothing; client EMA disabled to eliminate double-EMA (5x faster response) |
| `ema_alpha` | `0.5` | (unused while `action_smoothing=none`; sane default if manually re-enabled) |
| `smooth_gripper` | `false` | gripper uses hybrid mode discrete switching, no smoothing needed |
| `smooth_base` | `false` | base is covered by server-side ensemble + action_clip |

### 7. Stop the containers

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## Important Notes

### Root cause of R4 0/3 and R5 mitigation

R4 failed due to a **silent fallback** inside `PI05Policy.from_pretrained` (vision_tower load failures were caught with `except: print(...)` and replaced with random weights), causing the VLA to ignore the visual input.

R5 mitigations (all in place):
1. **lerobot fork @ramen `7431fb1d`**: silent fallback converted to `RuntimeError` + nested→flat auto-remap (PR #9)
2. **`PI05Policy.from_pretrained(strict=True)`**: explicitly set in `lerobot_hsr_policy.py`
3. **transformers 5.7.0**: flat SigLIPVisionModel matches Run72 ckpt 1:1
4. **bf16 quantization**: addresses VRAM constraint (RTX 5070 Ti 16 GB) — ~16.5 GB → ~9.9 GB

### Checkpoint preprocessing (automated by entrypoint.sh)

The following `config.json` fields are auto-fixed at startup (a working copy is used so read-only mounts still work):
- `compile_model: True → False` (true causes 300-sec timeout on first inference)
- `gradient_checkpointing: True → False` (unnecessary for inference; saves memory)
- DAFD fields removed: `use_dafd`, `dafd_gripper_*` × 6, `dafd_sign_*` × 2 (LeRobot compatibility; only present in Run72 training-time config)
- `policy_preprocessor.json` `tokenizer_name` rewritten to in-container path (offline-safe)

### State pad logic (for Run72 32D state ckpt)

The HSR client sends 8D state (`arm 5 + gripper 1 + head 2`), but Run72 ckpt's `policy_preprocessor.observation.state.{q01,q99,...}` is stored as 32D. `server/lerobot_hsr_policy.py::_pad_state_8d_to_32d` pads with the following layout:

| 8D src | 32D dst | Description |
|--------|---------|------|
| `[0:5]` | `[0:5]` | arm joints (arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll) |
| `[5]` | `[6]` | gripper (`hand_motor_joint`) |
| `[6:8]` | `[11:13]` | head pan / tilt |
| (none) | remaining 24 dims | 0 (base, etc.) |

The reverse action extraction (`_ACTION_32D_TO_11D`) uses indices `[0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]`.

### Action postprocessor (Issue #197 fixes, ported by PR #9)

`server/serve_hsr_policy_ws.py` wraps the policy with `ActionPostprocessor` for both e2e and HVLA modes, applying the `postprocessor` block in `hierarchical_config_optimized.yaml`:

```yaml
postprocessor:
  gripper_clip: true
  gripper_clip_min: -1.0
  gripper_clip_max: 1.239        # GT q99=1.2392 (HSR mechanical limit)
  gripper_binarize: false
  temporal_ensemble: true
  ensemble_window: 10            # ACT standard (matches chunk_size)
  ensemble_decay: 0.5
  action_smoothing: true
  action_smoothing_alpha: 0.5    # Server-only EMA; client side is off
  action_smoothing_exclude_dims: [5]  # gripper excluded from smoothing
  head_zero_mask: true
  head_dims: [6, 7]
  action_clip: true
  action_clip_ranges:
    10: [-0.32, 0.32]            # base_theta physical safety guard
  prompt_validation: true        # logs prompt format on real hardware
```

### About bf16 quantization

The R5 submission ckpt is **bf16 quantized**. fp32 inference requires ~16.5 GB VRAM, which OOMs on RTX 5070 Ti (16 GB). Hence we ship bf16 for deployment:

| Aspect | fp32 (training side) | **bf16 (R5 submission)** |
|---|---|---|
| `model.safetensors` | 15.4 GiB | **8.7 GiB** |
| Inference VRAM | ~16.5 GB | **~9.9 GB** |
| `config.json` `dtype` | `"float32"` | `"bfloat16"` |
| Inference latency (warm) | ~620 ms | **~370 ms** |
| Performance gap (offline eval, mean over 6 public tasks) | (baseline) | corr Δ -0.002, NBR Δ -0.024 (within tolerance; in fact slightly better) |

The quantization script lives at `icra_2026_ramen/eval/offline_evaluation/convert_ckpt_to_bf16.py`. The bf16 ckpt is also published to HF Hub as `pi05-round5-run73-r71s50-pf-noeval-32d-bf16`.

## Smoke Test

### Server-side log

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# Expected: "INFO:websockets.server:server listening on 0.0.0.0:8000"
```

### One-frame inference from host

```bash
PYTHONPATH=$(pwd)/packages/policy-client/src python3 -c "
import numpy as np
from policy_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host='127.0.0.1', port=8000)
print('metadata:', policy.get_server_metadata())

obs = {
    'head_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
    'hand_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
    'state': np.zeros(8, dtype=np.float32),
    'prompt': 'Pick up the box',
}
result = policy.infer(obs)
actions = np.asarray(result['actions'])
print(f'actions shape: {actions.shape}')         # expects (10, 11)
print(f'NaN: {np.isnan(actions).any()} / Inf: {np.isinf(actions).any()}')  # expects False/False
"
```

## Troubleshooting

| Symptom | Cause | Fix |
|------|------|------|
| `RuntimeError: ... Missing key(s) in state_dict: vision_tower.vision_model.*` | transformers still on 5.3.x (nested SigLIPVisionModel) | Verify `transformers==5.7.0` in `pyproject.toml`, then rebuild Docker |
| `RuntimeError: tensor a (8) ... b (32)` | state pad logic missing (older server code) | Confirm you are on `feat/lerobot-pi05-r5-run73` branch (`git log -1`) |
| `Warning: Could not load state dict` | lerobot pin pre-PR #9 | Verify `uv.lock` has `lerobot @ ramen 7431fb1d` or later |
| First inference takes 300+ sec | `compile_model: True` not stripped | Check entrypoint.sh auto-fix log lines |
| `RuntimeError: CUDA out of memory` | bf16 not effective (`config.dtype="float32"`) | `cat checkpoints/r5/config.json \| grep dtype` should show `"bfloat16"`. If fp32, re-download from HF Hub `pi05-round5-run73-r71s50-pf-noeval-32d-bf16` |
| Real-robot gripper response slow (~3 sec lag) | Double-EMA: client EMA (α=0.2) on top of server EMA (α=0.5) | Verify launch file has `action_smoothing="none"` (fixed in PR #9) |

## References

- LeRobot fork PR #9: https://github.com/matsuolab-llmcompe2025-team-suzuki/lerobot/pull/9 (vision_tower silent-fallback fix)
- airoa-evaluation-ICRA PR #4 (deploy server hardening + dummy camera removal)
- airoa-evaluation-ICRA PR #5 (transformers 5.3.0 → 5.7.0)
- airoa-evaluation-ICRA PR #7 (state 8D → 32D pad)
- airoa-evaluation-ICRA PR #8 (added this document)
- **airoa-evaluation-ICRA PR #9** (port of icra_2026_ramen PR #198: deploy bugs + safety guards)
- **airoa-evaluation-ICRA PR #10** (bf16 quantization support + this document update)
- icra_2026_ramen Issue #161 (lerobot pin / vision_tower bug)
- icra_2026_ramen Issue #193 / PR #194 (R5 deploy checklist)
- **icra_2026_ramen Issue #197 / PR #198** (Run72 deploy bugs + safety guards)
- icra_2026_ramen `eval/offline_evaluation/convert_ckpt_to_bf16.py` (bf16 quantization script)

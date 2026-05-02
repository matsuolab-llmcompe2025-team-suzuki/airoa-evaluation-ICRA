# airoa-evaluation-ICRA — Team RAMEN (Team 11)

Round 5 (Final) submission runtime for **ICRA 2026 VLA Workshop Competition**.

| Field | Value |
|---|---|
| **Branch** | `feat/lerobot-pi05-r5-run73` |
| **Backend** | LeRobot `PI05Policy` (e2e mode, single model) |
| **Checkpoint** | Run73 s20000 fine-tune of [`pi05-baseline-100k-pt`](https://huggingface.co/ICRA-2026-RAMEN/pi05-baseline-100k-pt) |
| **Quantization** | **bf16** (deploy-only, ~8.7 GiB) |
| **VRAM** | ~9.9 GB (RTX 5070 Ti 16 GB に余裕) |
| **Disk** | ~14.5 GB (Docker ~6.84 GB + ckpt ~8.7 GiB) |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05-r5-run73

# 2. Download checkpoint (R2)
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 sync \
    s3://airoa-icra-team-11/r5-pi05-run73-r71s50-pf-noeval-32d-s20000/ checkpoints/r5/

# 3. Start policy server (Docker)
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r5
./RUN-DOCKER-CONTAINER.sh up

# 4. Verify (sanity check)
docker logs airoa_policy_server 2>&1 | tail -20
# Expected:
#   "INFO:lerobot_hsr_policy:expected_state_dim=32 (HSR client sends 8D)"
#   "INFO:lerobot_hsr_policy:PI05Policy loaded successfully"
#   "INFO:websockets.server:server listening on 0.0.0.0:8000"

# 5. Enter HSR client shell + launch
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch

# 6. Stop
./RUN-DOCKER-CONTAINER.sh down
```

For full reproduction details:
- 🇯🇵 [R5_REPRODUCTION_STEPS_ja.md](R5_REPRODUCTION_STEPS_ja.md)
- 🇬🇧 [R5_REPRODUCTION_STEPS_en.md](R5_REPRODUCTION_STEPS_en.md)

## Architecture

```
Docker Container (airoa_policy_server)
└── LeRobotHSRPolicy           π0.5 fine-tuned, bf16, e2e mode
    ├── PI05Policy             transformers 5.7.0 + lerobot @ramen 7431fb1d
    ├── DataProcessorPipeline  state pad 8D→32D, normalize, tokenize, device transfer
    └── ActionPostprocessor    gripper_clip + temporal_ensemble + EMA + action_clip
        WebsocketPolicyServer  msgpack protocol, port 8000
```

Single-model e2e architecture. HSR client sends `(head_rgb, hand_rgb, state, prompt)`,
server returns `actions: (10, 11)` per inference.

## Configuration

All environment is pre-configured via `.env`:

| Variable | Value | Description |
|---|---|---|
| `POLICY_BACKEND` | `lerobot` | Backend (LeRobot PI05Policy) |
| `POLICY_MODE` | `e2e` | Inference mode (single-model, no HVLA) |
| `POLICY_SERVER_PORT` | `8000` | WebSocket server port |

Manual export required:
- `POLICY_CHECKPOINT_PATH` — absolute path to `checkpoints/r5/`

**HF_TOKEN is not required**. PaliGemma tokenizer is bundled in the container.

## Action Postprocessor (Issue #197 fixes)

`ActionPostprocessor` applies safety guards / smoothing on top of raw model outputs
(configured in `hierarchical_config_optimized.yaml`):

| Component | Setting | Purpose |
|---|---|---|
| `gripper_clip` | `[-1.0, 1.239]` | HSR mechanical limit (GT q99 = 1.2392) |
| `temporal_ensemble` | window=10, decay=0.5 | ACT-style chunk overlap (chunk_size=10) |
| `action_smoothing` | EMA α=0.5, gripper exclude | Server-side single EMA (response: 0.36s) |
| `head_zero_mask` | dims [6, 7] | GT 97%+ stationary, prevent unintended head motion |
| `action_clip` | dim 10 (base_theta) ±0.32 | Physical safety guard (HSR base ~1-2 rad/s) |
| `prompt_validation` | enabled | Logs prompt format for debugging |

Client-side EMA is **disabled** (`action_smoothing=none` in launch) to avoid
double-EMA response delay (5x slower).

## WebSocket I/O Contract

**Request**:
- `head_rgb`: `(H, W, 3)` uint8
- `hand_rgb`: `(H, W, 3)` uint8
- `state`: `(8,)` float32 (8D HSR layout: arm 5 + gripper + head 2)
- `prompt`: `str`

**Response**:
- `actions`: `(10, 11)` float32 (chunk_size=10, composite_11d action)

Action layout (composite_11d):
`[arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper, head_pan, head_tilt, base_x, base_y, base_theta]`

## Host Requirements

- Linux + Docker Engine + Docker Compose v2 + NVIDIA Container Toolkit
- NVIDIA GPU with **16 GB+ VRAM** (uses ~9.9 GB after bf16 quantization)
- Host RAM: **12 GB+** (new path `LEROBOT_LOW_CPU_MEM=1` default; old path needs ~40 GB)
- Host SSD: 20 GB+ (Docker image ~7 GB + ckpt ~8.7 GiB + working space)
- Internet (build-time only); inference works fully offline (`--network none` verified)

## Key Files

| File | Description |
|---|---|
| `R5_REPRODUCTION_STEPS_ja.md` / `_en.md` | Detailed reproduction (JA / EN) |
| `.env` | Pre-configured backend / mode / port |
| `docker-compose.yml` | Server + client container definitions |
| `RUN-DOCKER-CONTAINER.sh` | Convenience wrapper for `docker compose` |
| `server/Dockerfile` | CUDA 12.8.1 base (Blackwell compatible) |
| `server/entrypoint.sh` | Auto-fixes ckpt config (compile_model, DAFD fields, tokenizer path) |
| `server/serve_hsr_policy_ws.py` | WebSocket server entry point |
| `server/lerobot_hsr_policy.py` | LeRobot PI05Policy wrapper + state pad 8D→32D |
| `src/hierarchical_vla/action_postprocessor.py` | Action postprocessor (gripper_clip / EMA / action_clip / etc.) |
| `hierarchical_config_optimized.yaml` | Postprocessor + (optional HVLA) config |
| `deploy/hsr_policy_client/launch/hsr_policy_client.launch` | HSR client ROS launch |
| `tokenizer/paligemma-3b-pt-224/` | Bundled PaliGemma tokenizer (no HF_TOKEN needed) |
| `pa_decomposition_v2.json` | (HVLA mode only — unused in R5 e2e submission) |

## R4 → R5 Changes Summary

| Item | R4 | R5 |
|---|---|---|
| Model | run52 s040000 (8D output, fp32 ~8.8 GB) | **Run73 s20000 bf16** (32D output → 11D extract, 8.7 GiB) |
| Training data | `airoa-sft-v5` | **`airoa-public-filter-noeval`** (public-task focused) |
| `transformers` | 5.3.0 (nested SigLIPVisionModel) | **5.7.0** (flat SigLIPVisionModel) |
| `lerobot` fork | @ramen `c343490c` (vision_tower bug) | **@ramen `7431fb1d`** (PR #9 vision_tower fix) |
| `PI05Policy.from_pretrained` | `strict=False` (silent fallback risk) | **`strict=True`** (silent fallback prevented) |
| HSR state | 8D ckpt (no pad needed) | **8D → 32D pad** (`_pad_state_8d_to_32d`) |
| `config.json` `dtype` | `float32` | **`bfloat16`** (bf16 alloc) |
| Action postprocessor | basic clip + EMA | **gripper_clip 1.239, EMA α=0.5, ensemble window=10/decay=0.5, action_clip [base_theta ±0.32], prompt_validation** (Issue #197 safety guards) |
| Client EMA | `action_smoothing=ema, ema_alpha=0.2` | **`action_smoothing=none`** (double-EMA eliminated) |
| VRAM | ~9 GB | ~9.9 GB |
| CPU RAM peak | ~30 GB | **~9 GB** (PR #12 `LEROBOT_LOW_CPU_MEM=1` default, -78%) |
| Startup time | ~111 s | **~6 s** (PR #12 new path, -95%) |

## Submission Mode

R5 uses **e2e mode only** (`POLICY_MODE=e2e`, default). The hierarchical VLA pipeline
files (`hierarchical_*.py`, `pa_decomposition_v2.json`) remain in the repository for
reference but are not invoked during R5 evaluation.

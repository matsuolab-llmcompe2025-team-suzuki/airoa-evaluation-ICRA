# airoa-evaluation-ICRA — Team RAMEN (Team 11)

Round 5 (Final) submission runtime for **ICRA 2026 VLA Workshop Competition**.

| Field | Value |
|---|---|
| **Branch** | `feat/lerobot-pi05-r5` |
| **Backend** | LeRobot `PI05Policy` (e2e mode, single model) |
| **Checkpoint** | Run73 s020000 fine-tune of [`pi05-baseline-100k-pt`](https://huggingface.co/ICRA-2026-RAMEN/pi05-baseline-100k-pt) |
| **Quantization** | bf16 (deploy-only, ~9.35 GB) |
| **VRAM** | ~9.9 GB |
| **Disk** | ~16.5 GB (Docker ~7 GB + ckpt ~9.35 GB) |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05-r5

# 2. Download checkpoint (HF Hub, bf16)
huggingface-cli download \
    ICRA-2026-RAMEN/pi05-round5-run73-r71s50-pf-noeval-32d-s20000-bf16 \
    --local-dir checkpoints/r5

# 3. Start policy server (Docker)
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r5
./RUN-DOCKER-CONTAINER.sh up

# 4. Verify
docker logs airoa_policy_server 2>&1 | tail -5
# Expected: "INFO:websockets.server:server listening on 0.0.0.0:8000"

# 5. Enter HSR client shell + launch
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch

# 6. Stop
./RUN-DOCKER-CONTAINER.sh down
```

For full reproduction details:
- 🇯🇵 [R5_REPRODUCTION_STEPS_ja.md](R5_REPRODUCTION_STEPS_ja.md)
- 🇬🇧 [R5_REPRODUCTION_STEPS_en.md](R5_REPRODUCTION_STEPS_en.md)

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
- NVIDIA GPU with **16 GB+ VRAM**
- Host RAM: **12 GB+**
- Host SSD: 20 GB+ (Docker image ~7 GB + ckpt ~9.35 GB + working space)
- Internet (build-time only); inference works fully offline (`--network none` verified)

# R5 Reproduction Steps — Team RAMEN (Team 11)

## Overview

| Field | Value |
|------|-----|
| Model | π0.5 fine-tuned (Run73 s020000, bf16 quantized) |
| Checkpoint (R2) | `s3://airoa-icra-team-11/r5-pi05-run73-r71s50-pf-noeval-32d-s020000/` |
| Fork repository | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| Branch | `feat/lerobot-pi05-r5` |
| Backend | `lerobot` (pre-set in `.env`, no manual export needed) |
| Mode | `e2e` (pre-set in `.env`, no manual export needed) |
| VRAM | ~9.9 GB |
| SSD | Docker ~7 GB + checkpoint ~9.35 GB = ~16.5 GB (within 30 GB limit) |
| Tokenizer | Bundled in container (`/workspace/tokenizer/paligemma-3b-pt-224`) |

## Prerequisites

- NVIDIA GPU (16 GB+ VRAM)
- Docker + Docker Compose v2 + NVIDIA Container Toolkit
- HF_TOKEN is **not required** (PaliGemma tokenizer is bundled in the container)

## Reproduction Steps

### 1. Clone the repository

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05-r5
```

### 2. Download the checkpoint

```bash
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 sync \
    s3://airoa-icra-team-11/r5-pi05-run73-r71s50-pf-noeval-32d-s020000/ checkpoints/r5/
```

### 3. Start the container

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r5
./RUN-DOCKER-CONTAINER.sh up
```

Other env vars (`POLICY_BACKEND`, `POLICY_MODE`, etc.) are pre-set in `.env`. No manual export needed.

### 4. Connect to the HSR client container

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 5. Launch the HSR policy client (inside container)

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

### 6. Stop the containers

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## Checkpoint file layout

```
checkpoints/r5/
├── config.json                                                   (dtype="bfloat16")
├── model.safetensors                                             (~9.35 GB)
├── policy_preprocessor.json
├── policy_preprocessor_step_2_normalizer_processor.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json
```

## Smoke test

After the container starts, verify the policy server:

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# Expected: "INFO:websockets.server:server listening on 0.0.0.0:8000"
```

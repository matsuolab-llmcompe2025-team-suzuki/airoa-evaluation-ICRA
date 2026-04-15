# R4 Reproduction Steps — Team RAMEN (Team 11)

## Overview

| Item | Value |
|------|-------|
| Model | π0.5 fine-tuned SFT (run52-sft-v5) |
| Checkpoint (R2) | `s3://airoa-icra-team-11/pretrained_model/` |
| Repository | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| Branch | `feat/lerobot-pi05` |
| Backend | `lerobot` (set via `.env`, no manual export needed) |
| Mode | `e2e` (set via `.env`, no manual export needed) |
| VRAM | ~9 GB |
| SSD | Docker ~20 GB + checkpoint ~9 GB = ~29 GB (within 30 GB limit) |
| Tokenizer | Bundled in container (`/workspace/tokenizer/paligemma-3b-pt-224`) |

## Prerequisites

- NVIDIA GPU with 16 GB+ VRAM
- Host RAM: 16 GB+
- Docker + Docker Compose v2 + NVIDIA Container Toolkit
- AWS CLI (`pip install awscli`)
- No HF_TOKEN required (tokenizer is bundled in the container)

## Reproduction Steps

### 1. Clone the repository

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05
```

### 2. Download the checkpoint

```bash
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 sync \
    s3://airoa-icra-team-11/pretrained_model/ checkpoints/r4/
```

### 3. Start the container

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r4
./RUN-DOCKER-CONTAINER.sh up
```

All other environment variables (`POLICY_BACKEND`, `POLICY_MODE`, etc.) are pre-configured in the `.env` file. No manual `export` commands are needed.

### 4. Connect to the HSR client container

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 5. Launch the HSR policy client (inside the container)

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

### 6. Stop the containers

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## Important Notes

- `config.json` must have `"compile_model": false`. If set to `true`, the first inference will exceed the 300s timeout.
- HF_TOKEN is **not required**. The tokenizer (`google/paligemma-3b-pt-224`) is bundled in the Docker container at `/workspace/tokenizer/`.
- The `.env` file in the repository root sets all required environment variables automatically.

## Checkpoint Files

```
pretrained_model/
├── config.json              (compile_model=false)
├── model.safetensors        (~8.8 GB)
├── policy_preprocessor.json
├── policy_postprocessor.json
├── policy_preprocessor_step_2_normalizer_processor.safetensors
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json
```

## Verification

After starting the container, verify the policy server is running:

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# Expected: "server listening on 0.0.0.0:8000"
```

# R5 再現手順 — Team RAMEN (Team 11)

## 概要

| 項目 | 値 |
|------|-----|
| モデル | π0.5 ファインチューン (Run73 s020000) |
| HF Hub (deploy 用、 bf16) | `ICRA-2026-RAMEN/pi05-round5-run73-r71s50-pf-noeval-32d-s20000-bf16` |
| フォークリポジトリ | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| ブランチ | `feat/lerobot-pi05-r5` |
| バックエンド | `lerobot` (`.env` で設定済み、 手動 export 不要) |
| モード | `e2e` (`.env` で設定済み、 手動 export 不要) |
| VRAM | ~9.9 GB |
| SSD | Docker ~7 GB + チェックポイント ~9.35 GB = ~16.5 GB (30 GB 制限内) |
| トークナイザー | コンテナ内蔵 (`/workspace/tokenizer/paligemma-3b-pt-224`) |

## 前提条件

- NVIDIA GPU (16 GB+ VRAM)
- Docker + Docker Compose v2 + NVIDIA Container Toolkit
- `huggingface-cli` (`pip install huggingface_hub`)
- HF_TOKEN は **不要** (PaliGemma トークナイザーはコンテナに内蔵)

## 再現手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05-r5
```

### 2. チェックポイントのダウンロード

```bash
huggingface-cli download \
    ICRA-2026-RAMEN/pi05-round5-run73-r71s50-pf-noeval-32d-s20000-bf16 \
    --local-dir checkpoints/r5
```

### 3. コンテナの起動

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r5
./RUN-DOCKER-CONTAINER.sh up
```

その他の環境変数 (`POLICY_BACKEND`, `POLICY_MODE` 等) は `.env` ファイルで設定済みです。手動の `export` は不要です。

### 4. HSR クライアントコンテナに接続

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 5. HSR ポリシークライアントの起動 (コンテナ内)

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

### 6. コンテナの停止

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## チェックポイントのファイル構成

```
checkpoints/r5/
├── README.md
├── config.json                                                   (dtype="bfloat16")
├── model.safetensors                                             (~9.35 GB)
├── policy_preprocessor.json
├── policy_preprocessor_step_2_normalizer_processor.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json
```

## 動作確認

コンテナ起動後、 ポリシーサーバーの動作を確認:

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# 期待される出力: "INFO:websockets.server:server listening on 0.0.0.0:8000"
```

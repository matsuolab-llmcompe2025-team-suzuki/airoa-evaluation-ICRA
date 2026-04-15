# R4 再現手順 — Team RAMEN (Team 11)

## 概要

| 項目 | 値 |
|------|-----|
| モデル | π0.5 ファインチューン SFT (run52-sft-v5) |
| チェックポイント (R2) | `s3://airoa-icra-team-11/pretrained_model/` |
| フォークリポジトリ | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| ブランチ | `feat/lerobot-pi05` |
| バックエンド | `lerobot`（`.env` で設定済み、手動 export 不要） |
| モード | `e2e`（`.env` で設定済み、手動 export 不要） |
| VRAM | ~9 GB |
| SSD | Docker ~20 GB + チェックポイント ~9 GB = ~29 GB（30 GB 制限内） |
| トークナイザー | コンテナ内蔵 (`/workspace/tokenizer/paligemma-3b-pt-224`) |

## 前提条件

- NVIDIA GPU (16 GB+ VRAM)
- Docker + NVIDIA Container Toolkit
- HF_TOKEN は**不要**（トークナイザーはコンテナに内蔵）

## 再現手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05
```

### 2. チェックポイントのダウンロード

```bash
aws s3 sync s3://airoa-icra-team-11/pretrained_model/ checkpoints/r4/
```

### 3. コンテナの起動

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r4
./RUN-DOCKER-CONTAINER.sh up
```

その他の環境変数 (`POLICY_BACKEND`, `POLICY_MODE` 等) は `.env` ファイルで設定済みです。手動の `export` コマンドは不要です。

### 4. HSR クライアントコンテナに接続

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 5. HSR ポリシークライアントの起動（コンテナ内）

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

### 6. コンテナの停止

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## 重要な注意事項

- チェックポイントの `config.json` で `"compile_model": false` が**必須**。`true` の場合、初回推論が 300 秒を超えタイムアウトする
- HF_TOKEN は**不要**。トークナイザー (`google/paligemma-3b-pt-224`) は Docker コンテナに内蔵
- リポジトリルートの `.env` ファイルが全環境変数を自動設定する

## チェックポイントのファイル構成

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

## 動作確認

コンテナ起動後、ポリシーサーバーの動作を確認:

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# 期待される出力: "server listening on 0.0.0.0:8000"
```

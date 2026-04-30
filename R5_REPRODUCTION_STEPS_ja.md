# R5 再現手順 — Team RAMEN (Team 11)

## 概要

| 項目 | 値 |
|------|-----|
| モデル | π0.5 ファインチューン (Run72: `pf-noeval-32d`、aux-head 構造) |
| 提出 step | s029515 (final checkpoint) |
| ベースモデル | `ICRA-2026-RAMEN/pi05-baseline-100k-pt` (運営公開モデル) |
| 学習データ | `ICRA-2026-RAMEN/airoa-public-filter` |
| チェックポイント (R2) | `s3://airoa-icra-team-11/r5-pi05-run72-pf-noeval-32d-s029515/` |
| HF Hub (参考) | `ICRA-2026-RAMEN/pi05-round5-run72-pf-noeval-32d` @ commit `a7bbf6d4` (2026-04-29 19:59 UTC、step 029515 アップロード時) |
| フォークリポジトリ | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| ブランチ | `feat/lerobot-pi05-r5` |
| バックエンド | `lerobot`（`.env` で設定済み、手動 export 不要） |
| モード | `e2e`（`.env` で設定済み、HVLA 不使用、単一モデル方針） |
| VRAM | ~16.5 GB（Run72 32D output、A100 実測。RTX 5070 Ti 16 GB はギリギリ） |
| SSD | Docker ~6.84 GB + チェックポイント ~16 GB = **~22.8 GB**（30 GB 制限内、FP32 ckpt のまま） |
| トークナイザー | コンテナ内蔵 (`/workspace/tokenizer/paligemma-3b-pt-224`) |

## R5 における主要な変更点 (R4 との差分)

| 項目 | R4 | R5 |
|------|-----|-----|
| 提出モデル | run52 s040000 (8D 出力 → 11D pad) | **Run72 s029515** (32D 出力 → 11D 抽出) |
| 学習データ | `airoa-sft-v5` | **`airoa-public-filter`** (公開 task 特化) |
| transformers | 5.3.0 (nested SigLIPVisionModel) | **5.7.0** (flat SigLIPVisionModel) |
| lerobot fork | @ramen `c343490c` (vision_tower bug 含) | **@ramen `7431fb1d`** (PR #9 vision_tower fix 含) |
| `PI05Policy.from_pretrained` | デフォルト `strict=False` | **`strict=True`** (silent fallback を防止) |
| HSR state pad | (8D ckpt は pad 不要) | **8D → 32D pad 実装** (`_pad_state_8d_to_32d`) |
| VRAM | ~9 GB | ~16.5 GB |

## 前提条件

- NVIDIA GPU (**16 GB+ VRAM**、RTX 5070 Ti は要実測。A100 で 16.5 GB 観測)
- ホスト RAM: 16 GB 以上
- Docker (>= 20.10) + Docker Compose v2 + NVIDIA Container Toolkit
- AWS CLI (`pip install awscli`)
- ホスト SSD 空き: 30 GB 以上 (Docker image ~7 GB + ckpt ~16 GB + 作業領域)
- HF_TOKEN は**不要**（トークナイザーはコンテナに内蔵）
- インターネット接続: Docker build 時のみ必要、推論時は **`--network none` で動作可能**

## 再現手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/lerobot-pi05-r5
```

### 2. チェックポイントのダウンロード

```bash
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 sync \
    s3://airoa-icra-team-11/r5-pi05-run72-pf-noeval-32d-s029515/ checkpoints/r5/
```

期待されるファイル一覧:
```
checkpoints/r5/
├── config.json                                                 (3.4 KB)
├── model.safetensors                                           (~16 GB)
├── policy_preprocessor.json                                    (2.3 KB)
├── policy_preprocessor_step_2_normalizer_processor.safetensors (3.8 KB)
├── policy_postprocessor.json                                   (663 B)
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors (3.8 KB)
└── train_config.json                                           (8.0 KB)
```

### 3. コンテナの起動

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/r5
./RUN-DOCKER-CONTAINER.sh up
```

その他の環境変数 (`POLICY_BACKEND=lerobot`, `POLICY_MODE=e2e` 等) は `.env` ファイルで設定済みです。手動の `export` コマンドは不要です。

### 4. ポリシーサーバーの起動確認 (sanity check)

**重要**: R4 では `PI05Policy.from_pretrained` の silent bug が原因で実機評価 0/3 になりました。R5 では以下の確認を必ず行ってください。

```bash
# 起動完了 (最大 300 秒) を待ってから:
docker logs airoa_policy_server 2>&1 | tail -20
```

期待される出力（**全て満たすこと**）:
```
[entrypoint] Checkpoint copied to /workspace/_checkpoint
[entrypoint] config.json: compile_model True -> False
[entrypoint] config.json: gradient_checkpointing True -> False
[entrypoint] config.json: removed DAFD fields: ['use_dafd', 'dafd_gripper_dim', ...]
[entrypoint] policy_preprocessor.json: tokenizer_name google/paligemma-3b-pt-224 -> /workspace/tokenizer/paligemma-3b-pt-224
INFO:lerobot_hsr_policy:Loading PI05Policy from /workspace/_checkpoint on cuda
INFO:lerobot_hsr_policy:expected_state_dim=32 (HSR client sends 8D)         ← ★ 32D state pad 動作確認
INFO:lerobot_hsr_policy:PI05Policy loaded successfully                       ← ★ load 成功 (strict=True)
INFO:websockets.server:server listening on 0.0.0.0:8000                      ← ★ 起動完了
```

**もし以下のいずれかが出力された場合は、即停止して原因究明してください**:
- `Warning: Could not load state dict` → vision_tower load 失敗（lerobot pin が `7431fb1d` 以降になっているか要確認）
- `RuntimeError: ... Missing key(s) in state_dict` → PR #9 patch が効いている (silent fallback 防止)。transformers バージョンが 5.7.0 になっているか要確認
- `RuntimeError: tensor a (8) ... b (32)` → state pad が効いていない。本ブランチを使っているか要確認

### 5. HSR クライアントコンテナに接続

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 6. HSR ポリシークライアントの起動（コンテナ内）

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

主要 launch パラメータ (デフォルト):
- `gripper_mode:=hybrid` (R4 と同じ。close=力制御 (effort=-0.018)、open=連続位置)
- `prefetch_threshold:=0` (**RTC 無効、sync mode**。R4 と同じ。Issue #91 で「実機テスト時に `=30` 有効化検証推奨」と note されたが未検証のため判断保留。R5 は現状維持 = 0)
- `action_smoothing:=ema` / `ema_alpha:=0.2` (R4 と同じ)

### 7. コンテナの停止

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## 重要な注意事項

### R4 で発生した実機 0/3 の根本原因と R5 での対策

R4 では `PI05Policy.from_pretrained` 内に **silent fallback** (vision_tower の load 失敗を `except: print(...)` で握り潰して random 重みを返却) があり、画像を見ない VLA で実機評価が破綻しました (運営 4/24 結果)。

R5 では以下で完全に対策済:
1. **lerobot fork @ramen `7431fb1d`**: silent fallback を `RuntimeError` に変換 + nested→flat 自動 remap (PR #9)
2. **`PI05Policy.from_pretrained(strict=True)`**: `lerobot_hsr_policy.py` で明示
3. **transformers 5.7.0**: flat SigLIPVisionModel で Run72 ckpt と 1:1 一致

### チェックポイント前処理 (entrypoint.sh による自動化)

`config.json` の以下フィールドを起動時に自動修正します（read-only マウントでも動作するよう作業コピー使用）:
- `compile_model: True → False` (`true` だと初回推論 300 秒タイムアウト)
- `gradient_checkpointing: True → False` (推論不要、メモリ節約)
- DAFD フィールド除去: `use_dafd`, `dafd_gripper_*` ×6, `dafd_sign_*` ×2 (LeRobot 互換性、Run72 学習時のみ存在)
- `policy_preprocessor.json` の `tokenizer_name` をコンテナ内パスに書換 (オフライン対応)

### state pad ロジック (Run72 32D state ckpt 用)

HSR client が送る state は 8D (`arm 5 + gripper 1 + head 2`) ですが、Run72 ckpt の `policy_preprocessor.observation.state.{q01,q99,...}` は 32D で保存されています。`server/lerobot_hsr_policy.py` の `_pad_state_8d_to_32d` で以下の layout で pad:

| 8D src | 32D dst | 説明 |
|--------|---------|------|
| `[0:5]` | `[0:5]` | arm joints |
| `[5]` | `[6]` | gripper (`hand_motor_joint`) |
| `[6:8]` | `[11:13]` | head pan/tilt |
| (なし) | 残り 24 dim | 0 (base 等) |

逆方向の action 抽出 (`_ACTION_32D_TO_11D`) は `[0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]`。

## チェックポイントのファイル構成

```
pretrained_model/                                                       (合計 ~16 GB)
├── config.json                                                         (3.4 KB)
├── model.safetensors                                                   (~16 GB、32D output、aux-head 構造)
├── policy_preprocessor.json                                            (2.3 KB)
├── policy_preprocessor_step_2_normalizer_processor.safetensors         (3.8 KB、observation.state は 32D 保存)
├── policy_postprocessor.json                                           (663 B)
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors      (3.8 KB)
└── train_config.json                                                   (8.0 KB)
```

## 動作確認

コンテナ起動後、ポリシーサーバーの動作を確認:

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# 期待される出力: "INFO:websockets.server:server listening on 0.0.0.0:8000"
```

簡易動作確認 (host から WebSocket クライアントで 1 frame 推論):

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
    'prompt': 'pick the box',
}
result = policy.infer(obs)
actions = np.asarray(result['actions'])
print(f'actions shape: {actions.shape}')  # (10, 11) を期待
print(f'NaN: {np.isnan(actions).any()} / Inf: {np.isinf(actions).any()}')  # False/False を期待
"
```

## トラブルシューティング

| 症状 | 原因 | 対策 |
|------|------|------|
| `RuntimeError: ... Missing key(s) in state_dict: vision_tower.vision_model.*` | transformers が 5.3.x のまま (nested SigLIPVisionModel) | `pyproject.toml` の `transformers==5.7.0` 確認、Docker rebuild |
| `RuntimeError: tensor a (8) ... b (32)` | state pad ロジック未適用 (旧 server コード) | `feat/lerobot-pi05-r5` ブランチを使っているか確認 (`git log -1`) |
| `Warning: Could not load state dict` | lerobot pin が PR #9 以前 | `uv.lock` で `lerobot @ ramen 7431fb1d` 以降確認 |
| 初回推論が 300 秒以上 | `compile_model: True` のまま | entrypoint.sh の自動修正ログ確認 |
| `RuntimeError: CUDA out of memory` | RTX 5070 Ti (16 GB) で VRAM 不足 | A100 等 24 GB+ GPU 推奨。または FP16 変換検討 |

## 関連リソース

- LeRobot fork PR #9: https://github.com/matsuolab-llmcompe2025-team-suzuki/lerobot/pull/9
- airoa-evaluation-ICRA PR #5 (transformers 5.3.0 → 5.7.0)
- airoa-evaluation-ICRA PR #6 (state 8D → 32D pad)
- icra_2026_ramen Issue #161 (lerobot pin / vision_tower bug)
- icra_2026_ramen Issue #193 / PR #194 (R5 デプロイチェックリスト)

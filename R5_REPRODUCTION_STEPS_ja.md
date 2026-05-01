# R5 再現手順 — Team RAMEN (Team 11)

## 概要

| 項目 | 値 |
|------|-----|
| モデル | π0.5 ファインチューン (Run72: `pf-noeval-32d`、 32D output、 aux-head 構造) |
| 提出 step | s029515 (final checkpoint) |
| **量子化** | **bf16 量子化版** (deploy 用、 VRAM 制約対応) |
| ベースモデル | `ICRA-2026-RAMEN/pi05-baseline-100k-pt` (運営公開モデル) |
| 学習データ | `ICRA-2026-RAMEN/airoa-public-filter` |
| **チェックポイント (R2)** | `s3://airoa-icra-team-11/r5-pi05-run72-pf-noeval-32d-s029515/` (**bf16 量子化版、 7.7 GiB**) |
| **HF Hub (deploy 用、 bf16)** | **`ICRA-2026-RAMEN/pi05-round5-run72-pf-noeval-32d-bf16`** |
| HF Hub (学習側 fp32、 参考) | `ICRA-2026-RAMEN/pi05-round5-run72-pf-noeval-32d` @ commit `a7bbf6d4` |
| フォークリポジトリ | `https://github.com/matsuolab-llmcompe2025-team-suzuki/airoa-evaluation-ICRA` |
| ブランチ | `feat/lerobot-pi05-r5` |
| バックエンド | `lerobot` (`.env` で設定済み、 手動 export 不要) |
| モード | `e2e` (`.env` で設定済み、 HVLA 不使用、 単一モデル方針) |
| **VRAM** | **~9.7 GB** (bf16 量子化、 A100 実測。 RTX 5070 Ti 16 GB に余裕) |
| **SSD** | Docker ~6.84 GB + チェックポイント **~7.7 GiB** = **~14.5 GB** (30 GB 制限内) |
| トークナイザー | コンテナ内蔵 (`/workspace/tokenizer/paligemma-3b-pt-224`、 HF_TOKEN 不要) |

## R5 における主要な変更点 (R4 との差分)

| 項目 | R4 | R5 |
|------|-----|-----|
| 提出モデル | run52 s040000 (8D 出力 → 11D pad、 fp32 ~8.8 GB) | **Run72 s029515 bf16** (32D 出力 → 11D 抽出、 7.7 GiB) |
| 学習データ | `airoa-sft-v5` | **`airoa-public-filter`** (公開 task 特化) |
| `transformers` | 5.3.0 (nested SigLIPVisionModel) | **5.7.0** (flat SigLIPVisionModel) |
| `lerobot` fork | @ramen `c343490c` (vision_tower bug 含) | **@ramen `7431fb1d`** (PR #9 vision_tower fix) |
| `PI05Policy.from_pretrained` | デフォルト `strict=False` | **`strict=True`** (silent fallback 防止) |
| HSR state pad | (8D ckpt は pad 不要) | **8D → 32D pad 実装** (`_pad_state_8d_to_32d`) |
| `config.json` の `dtype` | `"float32"` | **`"bfloat16"`** (bf16 alloc に必須) |
| Action postprocessor | 簡易 EMA (α=0.3) + clip [-1.0, 1.23] | **Issue #197 修正一式** (下記表参照) |
| Client EMA | `action_smoothing=ema, ema_alpha=0.2` (二重 EMA) | **`action_smoothing=none`** (単独 server EMA、 5x 応答改善) |
| **VRAM** | ~9 GB | **~9.7 GB** (bf16 量子化、 fp32 16.5 GB → 9.7 GB) |
| **CPU RAM peak** | ~30 GB (PI05Policy.from_pretrained の二重持ち) | **~9 GB** (PR #12 新ルート `LEROBOT_LOW_CPU_MEM=1` default、 -78%) |
| **起動時間** | ~111 秒 | **~6 秒** (PR #12 新ルート、 -95%) |

### Issue #197 (PR #9) で導入された Action postprocessor 設定

`hierarchical_config_optimized.yaml` の `postprocessor` ブロック (e2e/HVLA 共通で適用):

| 機能 | 値 | 目的 |
|---|---|---|
| `gripper_clip` | `[-1.0, 1.239]` | HSR mechanical limit (GT q99=1.2392)、 R4 で 33 frames clip 解消 |
| `temporal_ensemble` | window=10, decay=0.5 | ACT (Zhao+2023) 流の chunk overlap (chunk_size=10 と整合) |
| `action_smoothing` | EMA α=0.5、 gripper exclude | 単独 server EMA で応答 5τ=0.36s (旧 0.3+client 0.2 の 5x 改善) |
| `head_zero_mask` | dims [6, 7] | 訓練 GT 97%+ が \|val\|<0.001 (head 静止)、 不要な首振り抑制 |
| `action_clip` | `dim 10 (base_theta) ±0.32` | 学習データ outlier (max=1.034 rad/step、 593 deg/s) を物理仕様内に safety guard |
| `prompt_validation` | true | 実機 prompt 形式を log で可視化 (動作変更なし) |

## 前提条件

- NVIDIA GPU (**16 GB VRAM 以上**で動作。 bf16 量子化により ~9.7 GB 使用)
- ホスト RAM: **12 GB 以上** (新ルート `LEROBOT_LOW_CPU_MEM=1` default で peak ~9 GB)
  - 旧ルート (`LEROBOT_LOW_CPU_MEM=0`) は CPU peak ~40 GB 必要、 24 GB 環境では OOM
- Docker (>= 20.10) + Docker Compose v2 + NVIDIA Container Toolkit
- AWS CLI (`pip install awscli`)
- ホスト SSD 空き: 20 GB 以上 (Docker image ~7 GB + ckpt ~7.7 GiB + 作業領域)
- HF_TOKEN は **不要** (PaliGemma トークナイザーはコンテナ内蔵)
- インターネット接続: Docker build 時のみ必要、 推論時は **`--network none` で動作可能**

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

期待されるファイル一覧 (bf16 量子化版):
```
checkpoints/r5/
├── config.json                                                 (2.9 KB、 dtype="bfloat16")
├── model.safetensors                                           (~7.7 GiB、 bf16 量子化済)
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

その他の環境変数 (`POLICY_BACKEND=lerobot`, `POLICY_MODE=e2e`, `POLICY_SERVER_PORT=8000`) は `.env` で設定済み、 手動 export 不要。

### 4. ポリシーサーバーの起動確認 (sanity check)

**重要**: R4 では `PI05Policy.from_pretrained` の silent fallback bug が原因で実機評価 0/3 になりました。 R5 では以下を必ず確認してください。

```bash
# 起動完了 (最大 300 秒) を待ってから:
docker logs airoa_policy_server 2>&1 | tail -20
```

期待される出力 (**全て満たすこと**):
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

**もし以下のいずれかが出力された場合は、 即停止して原因究明してください**:
- `Warning: Could not load state dict` → vision_tower load 失敗。 `uv.lock` で `lerobot @ ramen 7431fb1d` 以降を確認
- `RuntimeError: ... Missing key(s) in state_dict` → silent fallback 防止が効いている。 transformers が 5.7.0 か確認
- `RuntimeError: tensor a (8) ... b (32)` → state pad が効いていない。 `feat/lerobot-pi05-r5` ブランチを使っているか `git log -1` で確認
- `RuntimeError: CUDA out of memory` → bf16 化が効いていない。 `cat checkpoints/r5/config.json | grep dtype` で `"bfloat16"` を確認
- ホストプロセスが OOM Killed (CPU RAM 24 GB 環境等) → 新ルート (`LEROBOT_LOW_CPU_MEM=1` default) が無効化されている疑い。 `.env` で明示的に `=0` を設定していないか確認
- `Policy config not found: /workspace/hierarchical_config.yaml (using defaults)` → ActionPostprocessor wrap が **無効** で起動している (Issue #197 の safety guard 全て無効)。 `docker-compose.yml` の bind-mount が効いているか、 host 側に `hierarchical_config_optimized.yaml` が存在するか確認 (リポジトリ ルートに同梱済み)

### 5. HSR クライアントコンテナに接続

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

### 6. HSR ポリシークライアントの起動 (コンテナ内)

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

主要 launch パラメータ (デフォルト、 `deploy/hsr_policy_client/launch/hsr_policy_client.launch`):

| 引数 | デフォルト | 備考 |
|------|------|------|
| `gripper_mode` | `hybrid` | close=力制御 (effort=-0.018)、 open=連続位置 (R4 と同じ) |
| `prefetch_threshold` | `0` | RTC 無効、 sync mode (R4 と同じ) |
| `update_freq` | `10` | server 推論レート (Hz) |
| `adopted_action_chunks` | `10` | 1 chunk から採用する step 数 (model chunk_size=10 と一致) |
| `upsample` / `upsample_hz` / `upsample_method` | `true` / `100` / `spline` | 10Hz → 100Hz spline 補間 |
| **`action_smoothing`** | **`none`** | **PR #9 で `ema` から変更**。 server 側で EMA α=0.5 を実施するため client 側は OFF (二重 EMA を解消、 応答 5x 改善) |
| `ema_alpha` | `0.5` | (上記 OFF 時は未使用、 手動再有効化時の妥当値) |
| `smooth_gripper` | `false` | gripper は hybrid mode で discrete 切替するため smoothing 不要 |
| `smooth_base` | `false` | base は server 側 ensemble + action_clip でカバー |

### 7. コンテナの停止

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## 重要な注意事項

### R4 で発生した実機 0/3 の根本原因と R5 での対策

R4 では `PI05Policy.from_pretrained` 内に **silent fallback** (vision_tower の load 失敗を `except: print(...)` で握り潰して random 重みを返却) があり、 画像を見ない VLA で実機評価が破綻しました (運営 4/24 結果)。

R5 では以下で完全対策済:
1. **lerobot fork @ramen `7431fb1d`**: silent fallback を `RuntimeError` に変換 + nested→flat 自動 remap (PR #9)
2. **`PI05Policy.from_pretrained(strict=True)`**: `lerobot_hsr_policy.py` で明示
3. **transformers 5.7.0**: flat SigLIPVisionModel で Run72 ckpt と 1:1 一致
4. **bf16 量子化**: VRAM 制約 (RTX 5070 Ti 16 GB) に対応 (~16.5 GB → ~9.7 GB)

### チェックポイント前処理 (entrypoint.sh による自動化)

`config.json` の以下フィールドを起動時に自動修正します (read-only マウントでも動作するよう作業コピー使用):
- `compile_model: True → False` (true だと初回推論 300 秒タイムアウト)
- `gradient_checkpointing: True → False` (推論不要、 メモリ節約)
- DAFD フィールド除去: `use_dafd`, `dafd_gripper_*` × 6, `dafd_sign_*` × 2 (LeRobot 互換性)
- `policy_preprocessor.json` の `tokenizer_name` をコンテナ内パスに書換 (オフライン対応)

### state pad ロジック (Run72 32D state ckpt 用)

HSR client が送る state は 8D (`arm 5 + gripper 1 + head 2`) ですが、 Run72 ckpt の `policy_preprocessor.observation.state.{q01,q99,...}` は 32D で保存されています。 `server/lerobot_hsr_policy.py` の `_pad_state_8d_to_32d` で以下の layout で pad:

| 8D src | 32D dst | 説明 |
|--------|---------|------|
| `[0:5]` | `[0:5]` | arm joints (arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll) |
| `[5]` | `[6]` | gripper (`hand_motor_joint`) |
| `[6:8]` | `[11:13]` | head pan / tilt |
| (なし) | 残り 24 dim | 0 (base 等) |

逆方向の action 抽出 (`_ACTION_32D_TO_11D`) は `[0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]`。

### Action postprocessor (Issue #197 修正、 PR #9 で本 fork に反映)

`server/serve_hsr_policy_ws.py` で e2e mode でも `ActionPostprocessor` を wrap し、 `hierarchical_config_optimized.yaml` の `postprocessor` ブロック設定を適用:

```yaml
postprocessor:
  gripper_clip: true
  gripper_clip_min: -1.0
  gripper_clip_max: 1.239        # GT q99=1.2392 (HSR mechanical limit)
  gripper_binarize: false
  temporal_ensemble: true
  ensemble_window: 10            # ACT 標準 (chunk_size と一致)
  ensemble_decay: 0.5
  action_smoothing: true
  action_smoothing_alpha: 0.5    # 単独 server EMA、 client 側は OFF
  action_smoothing_exclude_dims: [5]  # gripper はスムージング除外
  head_zero_mask: true
  head_dims: [6, 7]
  action_clip: true
  action_clip_ranges:
    10: [-0.32, 0.32]            # base_theta 物理 safety guard
  prompt_validation: true        # 実機 prompt 形式 log
```

### bf16 量子化について

R5 提出 ckpt は **bf16 量子化版**。 fp32 のままだと推論時 VRAM ~16.5 GB 必要で RTX 5070 Ti (16 GB) で OOM するため、 deploy 用に bf16 化:

| 観点 | fp32 (元、 学習側) | **bf16 (R5 提出)** |
|---|---|---|
| `model.safetensors` | 15.4 GiB | **7.7 GiB** |
| 推論時 VRAM | ~16.5 GB | **~9.7 GB** |
| `config.json` の `dtype` | `"float32"` | `"bfloat16"` |
| 推論レイテンシ (warm) | ~620 ms | **~370 ms** |
| 性能差 (オフライン eval、 全 6 public task 平均) | (基準) | corr 差 -0.002、 NBR 差 -0.024 (実用範囲、 むしろ僅か改善) |

量子化方法は `icra_2026_ramen/eval/offline_evaluation/convert_ckpt_to_bf16.py` 参照。 HF Hub にも別 repo `pi05-round5-run72-pf-noeval-32d-bf16` として公開済。

## 動作確認

### サーバー側 ログ確認

```bash
docker logs airoa_policy_server 2>&1 | tail -5
# 期待される出力: "INFO:websockets.server:server listening on 0.0.0.0:8000"
```

### 簡易動作確認 (host から WebSocket クライアントで 1 frame 推論)

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
print(f'actions shape: {actions.shape}')         # (10, 11) を期待
print(f'NaN: {np.isnan(actions).any()} / Inf: {np.isinf(actions).any()}')  # False/False を期待
"
```

## トラブルシューティング

| 症状 | 原因 | 対策 |
|------|------|------|
| `RuntimeError: ... Missing key(s) in state_dict: vision_tower.vision_model.*` | transformers が 5.3.x のまま (nested SigLIPVisionModel) | `pyproject.toml` で `transformers==5.7.0` 確認、 Docker rebuild |
| `RuntimeError: tensor a (8) ... b (32)` | state pad ロジック未適用 (旧 server コード) | `feat/lerobot-pi05-r5` ブランチを使っているか `git log -1` で確認 |
| `Warning: Could not load state dict` | lerobot pin が PR #9 以前 | `uv.lock` で `lerobot @ ramen 7431fb1d` 以降を確認 |
| 初回推論が 300 秒以上 | `compile_model: True` のまま | entrypoint.sh の自動修正ログ確認 |
| `RuntimeError: CUDA out of memory` | bf16 化が効いていない (`config.dtype="float32"` のまま) | `cat checkpoints/r5/config.json \| grep dtype` で `"bfloat16"` を確認、 fp32 なら HF Hub `pi05-round5-run72-pf-noeval-32d-bf16` から再 download |
| 実機 gripper が反応遅い (~3 秒遅延) | client 側 EMA (α=0.2) と server 側 EMA (α=0.5) の二重 EMA | launch ファイルの `action_smoothing="none"` を確認 (PR #9 で修正済) |

## 関連リソース

- LeRobot fork PR #9: https://github.com/matsuolab-llmcompe2025-team-suzuki/lerobot/pull/9 (vision_tower silent-fallback fix)
- airoa-evaluation-ICRA PR #4 (deploy server hardening + dummy camera 削除)
- airoa-evaluation-ICRA PR #5 (transformers 5.3.0 → 5.7.0)
- airoa-evaluation-ICRA PR #7 (state 8D → 32D pad)
- airoa-evaluation-ICRA PR #8 (本ドキュメント追加)
- **airoa-evaluation-ICRA PR #9** (icra_2026_ramen PR #198 移植: deploy bugs + safety guards)
- **airoa-evaluation-ICRA PR #10** (bf16 量子化対応 + 本ドキュメント更新)
- icra_2026_ramen Issue #161 (lerobot pin / vision_tower bug)
- icra_2026_ramen Issue #193 / PR #194 (R5 デプロイチェックリスト)
- **icra_2026_ramen Issue #197 / PR #198** (Run72 deploy bugs + safety guards)
- icra_2026_ramen `eval/offline_evaluation/convert_ckpt_to_bf16.py` (bf16 量子化スクリプト)

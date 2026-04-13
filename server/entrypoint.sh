#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

# --- チェックポイント前処理 (モデル入替時の互換性を自動確保) ---
TOKENIZER_LOCAL="/workspace/tokenizer/paligemma-3b-pt-224"

/workspace/.venv/bin/python -c "
import json, sys, os

ckpt_dir = os.environ['POLICY_CHECKPOINT_DIR']
tokenizer_local = '${TOKENIZER_LOCAL}'

# 1. config.json: DAFD フィールドを除去 (LeRobot 互換性)
config_path = os.path.join(ckpt_dir, 'config.json')
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    dafd_keys = [k for k in config if 'dafd' in k.lower()]
    if dafd_keys:
        for k in dafd_keys:
            del config[k]
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f'[entrypoint] config.json: removed DAFD fields: {dafd_keys}')

# 2. policy_preprocessor.json: tokenizer_name をコンテナ内パスに書き換え (オフライン対応)
pp_path = os.path.join(ckpt_dir, 'policy_preprocessor.json')
if os.path.exists(pp_path) and os.path.isdir(tokenizer_local):
    with open(pp_path) as f:
        pp = json.load(f)
    changed = False
    for step in pp.get('steps', []):
        cfg = step.get('config', {})
        if 'tokenizer_name' in cfg and cfg['tokenizer_name'] != tokenizer_local:
            old = cfg['tokenizer_name']
            cfg['tokenizer_name'] = tokenizer_local
            changed = True
            print(f'[entrypoint] policy_preprocessor.json: tokenizer_name {old} -> {tokenizer_local}')
    if changed:
        with open(pp_path, 'w') as f:
            json.dump(pp, f, indent=2)
" || echo "[entrypoint] WARNING: checkpoint preprocessing failed (non-fatal)"

BACKEND="${POLICY_BACKEND:-openpi}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--backend" "${BACKEND}"
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

# --config-name is required for openpi, optional for lerobot
if [[ -n "${POLICY_CONFIG_NAME:-}" ]]; then
  ARGS+=("--config-name" "${POLICY_CONFIG_NAME}")
elif [[ "${BACKEND}" = "openpi" ]]; then
  echo "ERROR: POLICY_CONFIG_NAME is required for openpi backend" >&2
  exit 1
fi

if [[ -n "${POLICY_DEFAULT_PROMPT:-}" ]]; then
  ARGS+=("--default-prompt" "${POLICY_DEFAULT_PROMPT}")
fi

if [[ -n "${POLICY_RECORD_DIR:-}" ]]; then
  ARGS+=("--record-dir" "${POLICY_RECORD_DIR}")
fi

if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
  ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
fi

# HVLA モード
MODE="${POLICY_MODE:-e2e}"
if [[ "${MODE}" = "hierarchical" ]]; then
  ARGS+=("--mode" "hierarchical")
  ARGS+=("--pa-decomposition" "${PA_DECOMPOSITION:-/workspace/pa_decomposition.json}")
  ARGS+=("--policy-config" "${POLICY_CONFIG:-/workspace/hierarchical_config.yaml}")
  if [[ -n "${FM_MODEL:-}" ]]; then
    ARGS+=("--fm-model" "${FM_MODEL}")
  fi
  if [[ -n "${FM_SCALER:-}" ]]; then
    ARGS+=("--fm-scaler" "${FM_SCALER}")
  fi
  if [[ -n "${LLM_API_HOST:-}" ]]; then
    ARGS+=("--llm-api-host" "${LLM_API_HOST}")
  fi
  if [[ -n "${LLM_API_PORT:-}" ]]; then
    ARGS+=("--llm-api-port" "${LLM_API_PORT}")
  fi
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"

#!/usr/bin/env python3
"""Serve a policy as a WebSocket server for the HSR client.

Supports two backends:
  --backend openpi   (default) OpenPI framework (JAX/PyTorch, config-driven)
  --backend lerobot  LeRobot PI05Policy (merged checkpoint)

Supports two modes:
  --mode e2e          (default) End-to-end inference
  --mode hierarchical HVLA: PA decomposition + PA-level inference
"""
import argparse
import json
import logging
import os
from pathlib import Path

from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve policy as WebSocket server for HSR client")
    parser.add_argument("--backend", choices=["openpi", "lerobot"], default="openpi", help="Policy backend")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config-name", default=None, help="Train config name (required for openpi backend)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )
    # HVLA
    parser.add_argument("--mode", choices=["e2e", "hierarchical"], default="e2e",
                        help="Inference mode: e2e or hierarchical (HVLA)")
    parser.add_argument("--pa-decomposition", default="/workspace/pa_decomposition.json",
                        help="PA decomposition JSON file")
    parser.add_argument("--policy-config", default="/workspace/hierarchical_config.yaml",
                        help="HVLA config YAML file")
    parser.add_argument("--fm-model", default=None, help="FM RF model path (joblib)")
    parser.add_argument("--fm-scaler", default=None, help="FM StandardScaler path (joblib)")
    parser.add_argument("--llm-api-host", default="localhost", help="LLM API server host")
    parser.add_argument("--llm-api-port", type=int, default=8001, help="LLM API server port")
    return parser.parse_args()


def _create_openpi_policy(args):
    """Create policy using the OpenPI framework."""
    from openpi.policies import policy as policy_lib
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    if not args.config_name:
        raise ValueError("--config-name is required for openpi backend")

    config = train_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    return policy


def _create_lerobot_policy(args):
    """Create policy using LeRobot PI05Policy."""
    from lerobot_hsr_policy import LeRobotHSRPolicy

    device = args.pytorch_device or "cuda"
    return LeRobotHSRPolicy(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        default_prompt=args.default_prompt,
    )


def _sanitize_for_msgpack(obj):
    """Convert int dict keys to str recursively for msgpack strict_map_key=True compatibility.

    WebsocketClientPolicy (msgpack_numpy.unpackb) は strict_map_key=True で decode するため、
    metadata 内に int キーがあると `ValueError: int is not allowed for map key` になる。
    例: action_clip_ranges = {10: [-0.32, 0.32]} → {"10": [-0.32, 0.32]}
    """
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, int) else k): _sanitize_for_msgpack(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_for_msgpack(v) for v in obj]
    return obj


def _load_policy_config(args) -> dict:
    """Load policy_config YAML (used by both HVLA wrap and ActionPostprocessor wrap)."""
    import yaml

    if not os.path.exists(args.policy_config):
        logging.warning("Policy config not found: %s (using defaults)", args.policy_config)
        return {}
    with open(args.policy_config) as f:
        policy_cfg = yaml.safe_load(f) or {}
    logging.info("Policy config loaded: %s", args.policy_config)
    return policy_cfg


def _wrap_with_action_postprocessor(base_policy, policy_cfg: dict):
    """Wrap policy with ActionPostprocessor when any postprocessor flag is enabled.

    Issue #185 (icra_2026_ramen): action_smoothing 単独 ON config でも起動する。
    Issue #197 (icra_2026_ramen PR #198): base_deadband / gripper_closing_rate_cap /
    prompt_validation / action_clip を追加。
    R5 e2e モードでも有効化されることに注意 (config に postprocessor キーがあれば適用)。
    """
    pp_cfg = policy_cfg.get("postprocessor", {}) if policy_cfg else {}
    if not (
        pp_cfg.get("gripper_binarize") or pp_cfg.get("gripper_force_guard")
        or pp_cfg.get("gripper_contact_guard") or pp_cfg.get("temporal_ensemble")
        or pp_cfg.get("gripper_clip") or pp_cfg.get("action_smoothing")
        or pp_cfg.get("base_deadband") or pp_cfg.get("gripper_closing_rate_cap")
        or pp_cfg.get("prompt_validation") or pp_cfg.get("action_clip")
    ):
        return base_policy, None

    from hierarchical_vla.action_postprocessor import ActionPostprocessor

    wrapped = ActionPostprocessor(
        base_policy,
        gripper_clip=pp_cfg.get("gripper_clip", False),
        gripper_clip_min=pp_cfg.get("gripper_clip_min", 0.0),
        gripper_clip_max=pp_cfg.get("gripper_clip_max", 1.0),
        action_smoothing=pp_cfg.get("action_smoothing", False),
        action_smoothing_alpha=pp_cfg.get("action_smoothing_alpha", 0.3),
        action_smoothing_exclude_dims=pp_cfg.get("action_smoothing_exclude_dims", [5]),
        gripper_binarize=pp_cfg.get("gripper_binarize", False),
        gripper_threshold=pp_cfg.get("gripper_threshold", 0.5),
        gripper_idx=pp_cfg.get("gripper_idx", 5),
        gripper_force_guard=pp_cfg.get("gripper_force_guard", False),
        gripper_force_limit=pp_cfg.get("gripper_force_limit", 10.0),
        gripper_contact_guard=pp_cfg.get("gripper_contact_guard", False),
        gripper_contact_tolerance=pp_cfg.get("gripper_contact_tolerance", 0.005),
        gripper_contact_patience=pp_cfg.get("gripper_contact_patience", 3),
        temporal_ensemble=pp_cfg.get("temporal_ensemble", False),
        ensemble_window=pp_cfg.get("ensemble_window", 5),
        ensemble_decay=pp_cfg.get("ensemble_decay", 0.8),
        head_zero_mask=pp_cfg.get("head_zero_mask", False),
        head_dims=pp_cfg.get("head_dims", None),
        # Issue #197: B6 base deadband
        base_deadband=pp_cfg.get("base_deadband", False),
        base_dims=pp_cfg.get("base_dims", None),
        base_deadband_threshold=pp_cfg.get("base_deadband_threshold", 0.01),
        # Issue #197: B1' gripper closing rate cap
        gripper_closing_rate_cap=pp_cfg.get("gripper_closing_rate_cap", False),
        gripper_closing_rate_max=pp_cfg.get("gripper_closing_rate_max", 0.15),
        # Issue #197: B4 prompt validation
        prompt_validation=pp_cfg.get("prompt_validation", False),
        # Issue #197: action_clip (任意 dim 別 clip、 base_theta 物理 safety)
        action_clip=pp_cfg.get("action_clip", False),
        action_clip_ranges=pp_cfg.get("action_clip_ranges", None),
    )
    logging.info("ActionPostprocessor wrap enabled: %s", pp_cfg)
    return wrapped, pp_cfg


def _wrap_with_hvla(base_policy, args, policy_cfg: dict):
    """Wrap base policy with HVLA controller."""
    from hierarchical_hsr_policy import HierarchicalHSRPolicy

    # PA decomposition map
    with open(args.pa_decomposition) as f:
        pa_map = json.load(f)
    logging.info("PA map loaded: %d SHTs from %s", len(pa_map), args.pa_decomposition)

    # FM model
    fm_model = None
    fm_scaler = None
    if args.fm_model and os.path.exists(args.fm_model):
        try:
            import joblib
            fm_model = joblib.load(args.fm_model)
            if args.fm_scaler and os.path.exists(args.fm_scaler):
                fm_scaler = joblib.load(args.fm_scaler)
            logging.info("FM loaded: %s", args.fm_model)
        except Exception as e:
            logging.warning("FM load failed: %s", e)

    # LLM API クライアント（別プロセスの Qwen3.5-4B）
    llm_client = None
    try:
        from llm_api_client import LLMAPIClient
        llm_client = LLMAPIClient(
            host=args.llm_api_host,
            port=args.llm_api_port,
            pa_map=pa_map,
        )
        if llm_client.is_available():
            logging.info("LLM API client connected: %s:%d", args.llm_api_host, args.llm_api_port)
        else:
            logging.warning("LLM API server not available at %s:%d (PA マップ + E2E フォールバックで動作)",
                          args.llm_api_host, args.llm_api_port)
    except ImportError:
        logging.warning("LLM API client not available (llm_api_client.py not found)")

    hvla_policy = HierarchicalHSRPolicy(
        base_policy=base_policy,
        pa_map=pa_map,
        config=policy_cfg or {},
        fm_model=fm_model,
        fm_scaler=fm_scaler,
        llm_api_client=llm_client,
    )
    logging.info("HVLA mode: %d SHT, FM=%s, LLM=%s, Retry=enabled",
                 len(pa_map),
                 "enabled" if fm_model else "disabled",
                 "API" if (llm_client and llm_client.is_available()) else "disabled")
    return hvla_policy


def main() -> None:
    args = parse_args()

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    args.checkpoint_dir = checkpoint_dir

    if args.backend == "lerobot":
        policy = _create_lerobot_policy(args)
    else:
        policy = _create_openpi_policy(args)

    # Policy config を共通 load (e2e/HVLA 両方で使う、Issue #197 PR #198 対応)
    policy_cfg = _load_policy_config(args)

    # HVLA ラップ (mode=hierarchical 時のみ)
    if args.mode == "hierarchical":
        policy = _wrap_with_hvla(policy, args, policy_cfg)

    # ActionPostprocessor wrap (任意の postprocessor フラグが有効な場合、
    # Issue #185 / #197 PR #198。R5 e2e でも適用)
    policy, ap_cfg = _wrap_with_action_postprocessor(policy, policy_cfg)

    metadata = dict(policy.metadata) if hasattr(policy, "metadata") else {}
    metadata.update(
        {
            "backend": args.backend,
            "mode": args.mode,
            "config_name": args.config_name or "",
            "checkpoint_dir": checkpoint_dir,
            "server_host": args.host,
            "server_port": args.port,
        }
    )
    # action_clip_ranges 等の int キーを str に変換 (WebSocket client 側 msgpack 互換性)
    metadata = _sanitize_for_msgpack(metadata)

    logging.info(
        "Serving policy backend=%s mode=%s checkpoint=%s on %s:%s",
        args.backend, args.mode, checkpoint_dir, args.host, args.port,
    )
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

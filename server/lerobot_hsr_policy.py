#!/usr/bin/env python3
"""LeRobot PI05Policy wrapper for HSR WebSocket serving.

Wraps a merged LeRobot PI05Policy checkpoint to serve over the WebSocket
protocol expected by the AIRoA evaluation HSR client.

Camera mapping (training data → HSR client):
    hand_rgb  → observation.images.left_wrist_0_rgb
    head_rgb  → observation.images.base_0_rgb
    right_wrist_0_rgb, empty_camera_0 → dummy (-1.0)

Action padding:
    Model outputs 8D → padded to 11D (3 zeros for base_x, base_y, base_t)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from policy_client.base_policy import BasePolicy

logger = logging.getLogger(__name__)

_IMAGE_SIZE = (224, 224)
_HSR_ACTION_DIM = 11  # 8 joints + 3 base twist
_HSR_STATE_DIM = 8  # arm(5) + gripper(1) + head(2)

# 32D sparse layout → 11D composite のマッピング
# 出典: 公式 hsr_policy.py _decode_actions_inv の aligned_ids
# 学習側: model/scripts/data/preprocess.py remap_action_11d_to_32d (issue/110-baseline-ckpt-ft)
_ACTION_32D_TO_11D = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]


def _pad_state_8d_to_32d(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    """HSR 8D state を pi05_base layout に従って target_dim (typically 32) へゼロ pad。

    8D = [arm(5), gripper(1), head(2)]
    32D layout: arm=0:5, gripper=6, head=11:13, base=13:16, 残り 0
    eval/offline_evaluation/adapters/pi05_32d_adapter.py:328-339 と同等。
    """
    if state.shape[-1] >= target_dim:
        return state
    padded = torch.zeros(
        *state.shape[:-1], target_dim,
        dtype=state.dtype, device=state.device,
    )
    padded[..., 0:5] = state[..., 0:5]    # arm
    padded[..., 6] = state[..., 5]         # gripper
    padded[..., 11:13] = state[..., 6:8]   # head
    return padded


def _detect_state_dim(checkpoint_dir: str) -> int:
    """policy_preprocessor の observation.state stats shape から expected_state_dim を検出。

    eval/offline_evaluation/adapters/pi05_32d_adapter.py:_detect_state_dim と同等。
    検出失敗時は HSR 8D へフォールバック (旧 ckpt 互換)。
    """
    from safetensors import safe_open
    candidates = list(Path(checkpoint_dir).glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    for sf_path in candidates:
        try:
            with safe_open(str(sf_path), framework="pt") as f:
                keys = list(f.keys())
                for preferred in ("observation.state.q01", "observation.state.q99",
                                  "observation.state.mean", "observation.state.min"):
                    if preferred in keys:
                        return int(f.get_tensor(preferred).shape[0])
        except Exception as e:
            logger.warning("Failed to read %s for state dim detection: %s", sf_path, e)
    logger.warning("Could not detect expected state dim from preprocessor; falling back to %d", _HSR_STATE_DIM)
    return _HSR_STATE_DIM


class LeRobotHSRPolicy(BasePolicy):
    """LeRobot PI05Policy wrapper for HSR WebSocket serving."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda", default_prompt: str | None = None):
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        self._default_prompt = default_prompt or ""

        logger.info("Loading PI05Policy from %s on %s", checkpoint_dir, device)

        from lerobot.processor.pipeline import DataProcessorPipeline

        # MoE auto-detection: use PI05MoEPolicy if moe_config.json exists
        moe_config_path = Path(checkpoint_dir) / "moe_config.json"
        if moe_config_path.exists():
            logger.info("Detected moe_config.json → loading PI05MoEPolicy")
            from moe.policy import PI05MoEPolicy
            # MoE wraps PI05Policy; pass strict=True for silent-fallback protection (PR #9).
            try:
                self._policy = PI05MoEPolicy.from_pretrained(checkpoint_dir, strict=True)
            except TypeError:
                # Fallback for MoE classes that don't yet accept the strict kwarg.
                logger.warning("PI05MoEPolicy.from_pretrained does not accept strict=True; "
                               "loading without sanity check (silent-fallback risk)")
                self._policy = PI05MoEPolicy.from_pretrained(checkpoint_dir)
        else:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            # strict=True converts PR #9's silent-fallback (random vision_tower init)
            # into a hard RuntimeError. Without this we risk re-deploying R4's 0% bug.
            self._policy = PI05Policy.from_pretrained(checkpoint_dir, strict=True)

        self._policy.eval()
        self._policy.to(device)

        self._preprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir,
            "policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
        )
        self._postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir,
            "policy_postprocessor.json",
            overrides={"device_processor": {"device": device}},
        )

        # state pad target dim: preprocessor の observation.state stats から検出
        # 32D ckpt (Run60-72 等) は 32、8D ckpt (Run52 系) は 8
        self._expected_state_dim = _detect_state_dim(checkpoint_dir)
        logger.info("expected_state_dim=%d (HSR client sends %dD)",
                    self._expected_state_dim, _HSR_STATE_DIM)

        logger.info("PI05Policy loaded successfully")

    @property
    def supports_moe(self) -> bool:
        """Whether the loaded policy supports MoE expert selection."""
        return hasattr(self._policy, "select")

    def select_expert(self, instruction: str) -> int | None:
        """Select MoE expert based on instruction. Returns expert index or None if not MoE."""
        if not self.supports_moe:
            return None
        # Route only (no weight swap yet)
        idx = self._policy.model.router.route(instruction)
        # Skip redundant weight swap if same expert is already active
        if idx != self._policy.model.active_expert_idx:
            self._policy.model.select_expert(idx)
        return idx

    def infer(self, obs: dict) -> dict:
        """Convert HSR observation to LeRobot format, run inference, and return actions.

        Input obs keys:
            hand_rgb: [H, W, 3] uint8
            head_rgb: [H, W, 3] uint8
            state: [8] float
            prompt: str

        Returns:
            {"actions": np.ndarray [action_horizon, 11]}
        """
        hand_img = self._prepare_image(obs["hand_rgb"])
        head_img = self._prepare_image(obs["head_rgb"])

        prompt = obs.get("prompt", self._default_prompt) or self._default_prompt

        # Issue #168 (icra_2026_ramen): right_wrist_0_rgb / empty_camera_0 を
        # dummy=-1.0 で送信すると訓練 (LeRobot fork @ramen で missing keys mask=0)
        # と推論 (mask=1 + 値=-3.0 が vision_tower に流入) の非対称が発生し、
        # 評価結果が大幅に歪む。LeRobot の missing keys 分岐に委ねる。
        # HSR client sends 8D state; pad to expected_state_dim (32 for run60-72, 8 for run52 系).
        state = torch.tensor(np.asarray(obs["state"], dtype=np.float32), device=self._device)
        state = _pad_state_8d_to_32d(state, self._expected_state_dim)

        batch = {
            "observation.images.left_wrist_0_rgb": hand_img.unsqueeze(0),
            "observation.images.base_0_rgb": head_img.unsqueeze(0),
            "observation.state": state.unsqueeze(0),
            "task": prompt,
        }

        batch = self._preprocessor(batch)
        with torch.inference_mode():
            action = self._policy.predict_action_chunk(batch)
            # predict_action_chunk returns (B, chunk_size, action_dim).
            # Unpad to actual output dim. Safely access postprocessor metadata:
            # `features["action"]` may not exist for every step.
            original_dim = self._original_action_dim_or(action.shape[-1])
            if action.shape[-1] > original_dim:
                action = action[..., :original_dim]

        result = self._postprocessor({"action": action})

        action_out = result["action"].cpu().numpy()

        # Normalize ndim to (chunk_size, action_dim).
        if action_out.ndim == 3:
            # (B, chunk_size, action_dim)
            action_out = action_out[0]
        elif action_out.ndim == 1:
            # (action_dim,) — single step
            action_out = action_out[np.newaxis, :]
        elif action_out.ndim == 2:
            # Already (chunk_size, action_dim) — pass through.
            pass
        else:
            raise ValueError(
                f"Unexpected action ndim={action_out.ndim} shape={action_out.shape}; "
                "expected 1/2/3."
            )

        # 32D sparse layout → 11D composite (baseline-ft 等の 32D モデル対応).
        # Assert the source dim is exactly 32 to catch unexpected layouts (e.g.
        # 16D / 24D) before we reindex with stale `_ACTION_32D_TO_11D` indices.
        if action_out.shape[-1] > _HSR_ACTION_DIM:
            assert action_out.shape[-1] == 32, (
                f"32D→11D mapping requires action_dim==32, got {action_out.shape[-1]}. "
                "Did training pipeline change action layout?"
            )
            action_out = action_out[:, _ACTION_32D_TO_11D]
        # 11D 未満の場合はゼロパディング (旧 8D モデル対応)
        elif action_out.shape[-1] < _HSR_ACTION_DIM:
            pad_width = _HSR_ACTION_DIM - action_out.shape[-1]
            action_out = np.pad(action_out, ((0, 0), (0, pad_width)), mode="constant")

        return {"actions": action_out}

    def _original_action_dim_or(self, default: int) -> int:
        """Return the postprocessor's expected action dim, falling back safely.

        `self._postprocessor.steps[0].features["action"]` is not guaranteed to
        exist on every pipeline step (e.g. plain device_processor steps lack
        `features`); accessing it directly raises KeyError or AttributeError.
        Walk the steps and return the first valid features.action.shape[0].
        """
        for step in self._postprocessor.steps:
            features = getattr(step, "features", None)
            if features is None:
                continue
            feat = features.get("action") if hasattr(features, "get") else None
            if feat is None:
                continue
            shape = getattr(feat, "shape", None)
            if shape:
                return int(shape[0])
        return int(default)

    def reset(self) -> None:
        self._policy.reset()

    def _prepare_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert raw image to [3, 224, 224] float tensor."""
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize(_IMAGE_SIZE, Image.Resampling.BICUBIC)
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).to(self._device)

    @property
    def metadata(self) -> dict:
        return {
            "backend": "lerobot",
            "checkpoint_dir": self._checkpoint_dir,
        }

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

# 32D sparse layout → 11D composite のマッピング
# 出典: 公式 hsr_policy.py _decode_actions_inv の aligned_ids
# 学習側: model/scripts/data/preprocess.py remap_action_11d_to_32d (issue/110-baseline-ckpt-ft)
_ACTION_32D_TO_11D = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]


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
        batch = {
            "observation.images.left_wrist_0_rgb": hand_img.unsqueeze(0),
            "observation.images.base_0_rgb": head_img.unsqueeze(0),
            "observation.state": torch.tensor(
                np.asarray(obs["state"], dtype=np.float32),
                device=self._device,
            ).unsqueeze(0),
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

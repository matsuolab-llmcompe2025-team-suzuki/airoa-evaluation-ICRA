"""アクション後処理: グリッパー制御ガード + テンポラルアンサンブル

任意の BasePolicy をラップし、出力アクションに後処理を適用する。
Hierarchical VLA / E2E 両モードで使用可能。

グリッパー保護機構:
  1. 力覚ガード: wrist_wrench の力ノルムが上限超過時、閉じ方向をブロック
  2. 接触検知ガード: 閉じ指令を出しても位置が動かない場合、接触と判定して閉じ方向をブロック
     （追加データ購読不要。observation.state のみで動作）
"""

import logging
from collections import deque
from typing import Dict

import numpy as np
from policy_client.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ActionPostprocessor(BasePolicy):
    """アクション後処理ラッパー

    Args:
        policy: 内部ポリシー（BasePolicy 互換）
        gripper_binarize: グリッパー二値化を有効にする
        gripper_threshold: 二値化閾値（これ以上で開=1.0、未満で閉=0.0）
        gripper_idx: アクション配列中のグリッパーインデックス
        gripper_force_guard: 力覚センサー閾値ガードを有効にする
        gripper_force_limit: 把持力上限 [N]（超過時にグリッパー指令を現在値に固定）
        gripper_contact_guard: 接触検知ガードを有効にする
        gripper_contact_tolerance: 位置変化の閾値（これ未満で「動いていない」と判定）
        gripper_contact_patience: 連続何ステップ動かなければ接触と判定するか
        temporal_ensemble: テンポラルアンサンブルを有効にする
        ensemble_window: アンサンブル窓幅（過去 N ステップ）
        ensemble_decay: 指数減衰率（0〜1、1 に近いほど直近を重視）
    """

    def __init__(
        self,
        policy: BasePolicy,
        *,
        gripper_binarize: bool = True,
        gripper_threshold: float = 0.5,
        gripper_idx: int = 5,
        gripper_force_guard: bool = False,
        gripper_force_limit: float = 10.0,
        gripper_contact_guard: bool = False,
        gripper_contact_tolerance: float = 0.005,
        gripper_contact_patience: int = 3,
        gripper_clip: bool = False,
        gripper_clip_min: float = 0.0,
        gripper_clip_max: float = 1.0,
        action_smoothing: bool = False,
        action_smoothing_alpha: float = 0.3,
        action_smoothing_exclude_dims: list[int] | None = None,
        temporal_ensemble: bool = False,
        ensemble_window: int = 5,
        ensemble_decay: float = 0.8,
        head_zero_mask: bool = False,
        head_dims: list[int] | None = None,
        # Issue #197: B6 base deadband (probe v6 で 4/6 task の base_theta NBR>1.0 を発見)
        base_deadband: bool = False,
        base_dims: list[int] | None = None,
        base_deadband_threshold: float = 0.01,
        # Issue #197: action_clip (任意 dim 別 clip、 物理 safety guard)
        # 学習データに base_theta=1.034 rad/step (= 593 deg/s) outlier 含有を確認。
        # HSR base 物理仕様 (~1-2 rad/s) 超過 → deploy 側 clip で保護
        action_clip: bool = False,
        action_clip_ranges: dict[int, list[float]] | None = None,
        # Issue #197: B1' gripper closing rate cap (probe v3 で 5-9 frame 早閉じ確認)
        gripper_closing_rate_cap: bool = False,
        gripper_closing_rate_max: float = 0.15,
        # Issue #197: B4 prompt validation logging
        prompt_validation: bool = False,
    ):
        self._policy = policy
        self._gripper_binarize = gripper_binarize
        self._gripper_threshold = gripper_threshold
        self._gripper_idx = gripper_idx
        self._gripper_clip = gripper_clip
        self._gripper_clip_min = gripper_clip_min
        self._gripper_clip_max = gripper_clip_max
        self._action_smoothing = action_smoothing
        self._action_smoothing_alpha = action_smoothing_alpha
        self._action_smoothing_exclude = action_smoothing_exclude_dims or [5]  # default: gripper 除外
        self._head_zero_mask = head_zero_mask
        # Issue #187: composite_11d 規約で head_pan=6, head_tilt=7
        self._head_dims = head_dims if head_dims is not None else [6, 7]
        self._prev_action: np.ndarray | None = None
        self._gripper_force_guard = gripper_force_guard
        self._gripper_force_limit = gripper_force_limit
        self._gripper_contact_guard = gripper_contact_guard
        self._gripper_contact_tolerance = gripper_contact_tolerance
        self._gripper_contact_patience = gripper_contact_patience
        self._last_gripper_value: float | None = None
        self._gripper_stall_count: int = 0
        self._last_gripper_state: float | None = None
        self._contact_detected: bool = False
        self._temporal_ensemble = temporal_ensemble
        self._ensemble_window = ensemble_window
        self._ensemble_decay = ensemble_decay

        # Issue #197: base deadband (composite_11d の base_x=8, base_y=9, base_theta=10)
        self._base_deadband = base_deadband
        self._base_dims = base_dims if base_dims is not None else [8, 9, 10]
        self._base_deadband_threshold = base_deadband_threshold

        # Issue #197: action_clip (任意 dim 別 clip)
        # 訓練データ outlier 確認: base_theta max=1.034 rad/step (= 593 deg/s)
        # HSR base 物理仕様 ~1-2 rad/s 超過、 deploy 側で物理 safety guard 必要
        # GT preprocessor stats q99=±0.306 を 1.05x マージンで ±0.32 推奨
        self._action_clip = action_clip
        # config から dict[int, list] が来るが、 内部では dict[int, tuple] で保持
        self._action_clip_ranges: dict[int, tuple[float, float]] = {}
        if action_clip_ranges:
            for k, v in action_clip_ranges.items():
                self._action_clip_ranges[int(k)] = (float(v[0]), float(v[1]))

        # Issue #197: gripper closing rate cap
        self._gripper_closing_rate_cap = gripper_closing_rate_cap
        self._gripper_closing_rate_max = gripper_closing_rate_max
        self._last_gripper_cmd: float | None = None

        # Issue #197: prompt validation (実機 prompt が想定通り渡っているかを log)
        self._prompt_validation = prompt_validation
        self._last_logged_prompt: str | None = None

        # アンサンブルバッファ（各要素は 1D アクション配列）
        self._action_buffer: deque[np.ndarray] = deque(maxlen=ensemble_window)

        # 重み（指数減衰: 最新=1, 1つ前=decay, 2つ前=decay^2, ...）
        self._weights = np.array(
            [ensemble_decay ** i for i in range(ensemble_window)],
            dtype=np.float64,
        )

    @property
    def metadata(self) -> dict:
        inner_meta = getattr(self._policy, "metadata", {})
        return {
            **inner_meta,
            "postprocessor": {
                "gripper_binarize": self._gripper_binarize,
                "gripper_threshold": self._gripper_threshold,
                "gripper_force_guard": self._gripper_force_guard,
                "gripper_force_limit": self._gripper_force_limit,
                "gripper_contact_guard": self._gripper_contact_guard,
                "gripper_contact_tolerance": self._gripper_contact_tolerance,
                "gripper_contact_patience": self._gripper_contact_patience,
                "gripper_clip": self._gripper_clip,
                "gripper_clip_min": self._gripper_clip_min,
                "gripper_clip_max": self._gripper_clip_max,
                "action_smoothing": self._action_smoothing,
                "action_smoothing_alpha": self._action_smoothing_alpha,
                "temporal_ensemble": self._temporal_ensemble,
                "ensemble_window": self._ensemble_window,
                "ensemble_decay": self._ensemble_decay,
                "head_zero_mask": self._head_zero_mask,
                "head_dims": self._head_dims,
                "base_deadband": self._base_deadband,
                "base_dims": self._base_dims,
                "base_deadband_threshold": self._base_deadband_threshold,
                "gripper_closing_rate_cap": self._gripper_closing_rate_cap,
                "gripper_closing_rate_max": self._gripper_closing_rate_max,
                "prompt_validation": self._prompt_validation,
                "action_clip": self._action_clip,
                "action_clip_ranges": {k: list(v) for k, v in self._action_clip_ranges.items()},
            },
        }

    def infer(self, obs: Dict) -> Dict:
        # Issue #197: B4 prompt validation log
        if self._prompt_validation:
            self._validate_prompt(obs)

        result = self._policy.infer(obs)
        actions = result.get("actions")
        if actions is None:
            return result

        actions = np.asarray(actions, dtype=np.float64)

        # テンポラルアンサンブル（アクション horizon の先頭ステップに適用）
        if self._temporal_ensemble and actions.ndim >= 1:
            # horizon がある場合は先頭行、1D ならそのまま
            first_action = actions[0] if actions.ndim == 2 else actions
            self._action_buffer.appendleft(first_action.copy())
            first_action = self._compute_ensemble()
            if actions.ndim == 2:
                actions[0] = first_action
            else:
                actions = first_action

        # グリッパークリッピング（二値化の前に適用）
        if self._gripper_clip:
            actions = self._clip_gripper(actions)

        # Action Smoothing (EMA, gripper 除外)
        if self._action_smoothing:
            actions = self._smooth_action(actions)

        # Issue #197: B1' gripper closing rate cap (smoothing と binarize の間に適用)
        if self._gripper_closing_rate_cap:
            actions = self._apply_gripper_closing_rate_cap(actions)

        # Issue #197: B6 base deadband (base 系の微小ノイズを 0 に)
        if self._base_deadband:
            actions = self._apply_base_deadband(actions)

        # Issue #197: action_clip (任意 dim 別 clip、 base_theta 等の物理 safety guard)
        if self._action_clip:
            actions = self._apply_action_clip(actions)

        # グリッパー二値化
        if self._gripper_binarize:
            actions = self._binarize_gripper(actions)

        # 力覚センサー閾値ガード
        if self._gripper_force_guard:
            actions = self._apply_force_guard(actions, obs)

        # 接触検知ガード（observation.state ベース）
        if self._gripper_contact_guard:
            actions = self._apply_contact_guard(actions, obs)

        # Issue #187: head action のゼロマスク (GT 97%+ が |val|<0.001)
        if self._head_zero_mask:
            actions = self._apply_head_zero_mask(actions)

        result["actions"] = actions
        return result

    def reset(self) -> None:
        """ポリシーとアンサンブルバッファをリセット"""
        self._action_buffer.clear()
        self._last_gripper_value = None
        self._gripper_stall_count = 0
        self._last_gripper_state = None
        self._contact_detected = False
        self._prev_action = None
        self._last_gripper_cmd = None
        self._last_logged_prompt = None
        self._policy.reset()

    def reset_ensemble_buffer(self) -> None:
        """PA 切替時など、アンサンブルバッファのみリセット (legacy、reset_pa 推奨)"""
        self._action_buffer.clear()

    def reset_pa(self) -> None:
        """PA 切替時のリセット。policy 自体の reset は呼ばない。

        Issue #189: PA を跨ぐべきでない後処理の transient state を全てクリアする。
        - ensemble_buffer:    chunk 境界の重み付き平均
        - prev_action:        action_smoothing の EMA 直前値
        - last_gripper_value: force_guard が直前 gripper 指令を比較に使う
        - last_gripper_state: contact_guard が直前 gripper 実位置を保持
        - gripper_stall_count / contact_detected: contact_guard 状態機

        これらが残留すると、新 PA 開始直後に前 PA 終端値が混入し、
        smoothing で動き出しが鈍る / force_guard が誤発火 / contact_detected が
        前 PA から True のまま閉じ指令を即ブロックする等の transient bug を起こす。
        """
        self._action_buffer.clear()
        self._prev_action = None
        self._last_gripper_value = None
        self._last_gripper_state = None
        self._gripper_stall_count = 0
        self._contact_detected = False
        # Issue #197: gripper rate cap も PA 跨ぎでリセット (前 PA 終端値からの cap 残留を防ぐ)
        self._last_gripper_cmd = None

    def process_action(self, actions: np.ndarray, obs: dict | None = None) -> np.ndarray:
        """アクションに後処理を適用する（ポリシーを経由せず直接呼び出し可能）。

        infer() と同じ処理チェーンを適用するが、ポリシー推論は行わない。
        シミュレータサーバー等、ポリシーが別管理の場合に使用。

        Args:
            actions: (action_dim,) or (T, action_dim) のアクション配列
            obs: 観測データ（force_guard/contact_guard で使用。不要なら None）

        Returns:
            後処理済みアクション
        """
        actions = np.asarray(actions, dtype=np.float64)
        if obs is None:
            obs = {}

        if self._temporal_ensemble and actions.ndim >= 1:
            first_action = actions[0] if actions.ndim == 2 else actions
            self._action_buffer.appendleft(first_action.copy())
            first_action = self._compute_ensemble()
            if actions.ndim == 2:
                actions[0] = first_action
            else:
                actions = first_action

        if self._gripper_clip:
            actions = self._clip_gripper(actions)

        if self._action_smoothing:
            actions = self._smooth_action(actions)

        # Issue #197: B1' / B6
        if self._gripper_closing_rate_cap:
            actions = self._apply_gripper_closing_rate_cap(actions)
        if self._base_deadband:
            actions = self._apply_base_deadband(actions)
        if self._action_clip:
            actions = self._apply_action_clip(actions)

        if self._gripper_binarize:
            actions = self._binarize_gripper(actions)

        if self._gripper_force_guard:
            actions = self._apply_force_guard(actions, obs)

        if self._gripper_contact_guard:
            actions = self._apply_contact_guard(actions, obs)

        # Issue #187: head action のゼロマスク
        if self._head_zero_mask:
            actions = self._apply_head_zero_mask(actions)

        return actions

    def _apply_action_clip(self, actions: np.ndarray) -> np.ndarray:
        """Issue #197: 任意 dim 別の action clip (物理 safety guard)。

        訓練データ調査 (airoa-public-filter) で base_theta に max=1.034 rad/step
        (= 593 deg/s) outlier が含まれることを確認。 HSR base 物理仕様 (~1-2 rad/s)
        を 5-10x 超過。 model はこれを学習し、 deploy で同様の値を出力する可能性。

        deploy 側 clip で物理限界を保護する safety guard。 学習側 outlier 除去
        (= 真の根本解決) は再学習が必要なため、 R5 時点では deploy clip で対処。

        gripper_clip との違い: action_clip は任意 dim を clip 可能 (gripper 専用でない)。
        config 例:
            action_clip: true
            action_clip_ranges:
              10: [-0.32, 0.32]   # base_theta: GT q99=±0.306 を 1.05x マージン

        Args:
            actions: (D,) または (T, D) のアクション配列
        """
        for dim, (lo, hi) in self._action_clip_ranges.items():
            if actions.ndim == 2:
                if dim < actions.shape[1]:
                    actions[:, dim] = np.clip(actions[:, dim], lo, hi)
            elif actions.ndim == 1:
                if dim < len(actions):
                    actions[dim] = float(np.clip(actions[dim], lo, hi))
        return actions

    def _apply_base_deadband(self, actions: np.ndarray) -> np.ndarray:
        """Issue #197 (B6): base 系 (composite_11d dim 8/9/10) の微小ノイズを 0 に。

        probe v6 の発見: pick-coffee 等 6 task 中 4 task で base_theta NBR>1.0
        (model が GT=0 のところで ±0.07 のノイズを出している)。

        実機影響: base が無駄に動くと視覚 feedback ループが崩れ、 把持位置がずれる。

        principle: GT base velocity の std はタスク全体で <0.02 程度。
        |cmd| < threshold は信号でなくノイズと見なし 0 にする (deadband filter)。

        Args:
            actions: (D,) または (T, D) のアクション
        """
        for dim in self._base_dims:
            if actions.ndim == 2:
                if dim < actions.shape[1]:
                    mask = np.abs(actions[:, dim]) < self._base_deadband_threshold
                    actions[mask, dim] = 0.0
            elif actions.ndim == 1:
                if dim < len(actions) and abs(actions[dim]) < self._base_deadband_threshold:
                    actions[dim] = 0.0
        return actions

    def _apply_gripper_closing_rate_cap(self, actions: np.ndarray) -> np.ndarray:
        """Issue #197 (B1'): gripper closing rate を 1 step あたり最大 N に制限。

        probe v3/v4 の発見: pick-coffee で gripper closing が GT より 5-9 frame 早い。
        smoothing α=0.3 でも 6 frame 早閉じが残る (chunk-level の future prediction bias)。

        principle: 物理的には gripper の closing 速度は機械的限界がある。
        指令側で「1 step に変化できる closing rate」を上限を設けることで、
        早閉じ予測を「滑らかに伸ばし」ながら timing を遅延させる。

        composite_11d 規約: 大=open, 小=close。
        closing = cmd が prev より下がる方向。

        cap 適用条件: cmd < prev かつ |cmd - prev| > rate_max
        → cmd = prev - rate_max (close 方向の変化量を制限)

        Open 方向 (cmd > prev) は制限しない (release は速い方が good)。
        """
        idx = self._gripper_idx
        if actions.ndim == 2:
            current_cmd = float(actions[0, idx])
        elif actions.ndim == 1:
            current_cmd = float(actions[idx])
        else:
            return actions

        if self._last_gripper_cmd is not None:
            delta = current_cmd - self._last_gripper_cmd
            # 閉じ方向 (delta < 0) で変化量が cap を超えていたら制限
            if delta < 0 and abs(delta) > self._gripper_closing_rate_max:
                capped = self._last_gripper_cmd - self._gripper_closing_rate_max
                logger.debug(
                    "gripper closing rate cap: %.3f → %.3f (delta=%.3f, cap=%.3f)",
                    current_cmd, capped, delta, self._gripper_closing_rate_max,
                )
                if actions.ndim == 2:
                    actions[:, idx] = capped
                else:
                    actions[idx] = capped
                current_cmd = capped

        self._last_gripper_cmd = current_cmd
        return actions

    def _validate_prompt(self, obs: Dict) -> None:
        """Issue #197 (B4): obs に渡される prompt の形式を log で可視化。

        probe v4 の発見: 英語 prompt 間で動作差が小さい (|Δ|≈0.01-0.02) が、
        empty / 日本語では action が激変 (|Δ|=0.10)。
        → 実機 prompt が想定通りに英語で渡っているかの不確実性。

        本メソッドは prompt の長さ・先頭文字種・変化を log するだけで、
        action 値は変更しない (pure observability)。

        対応 obs key: "task" / "prompt" (どちらか先に存在する方)
        """
        # 注: empty string ("") も検出対象なので `or` ではなく明示的に None チェック
        if "task" in obs:
            prompt = obs["task"]
        elif "prompt" in obs:
            prompt = obs["prompt"]
        else:
            return
        if prompt is None:
            return

        prompt_str = str(prompt)
        if prompt_str == self._last_logged_prompt:
            return  # 同じ prompt は重複 log しない

        is_ascii = all(ord(c) < 128 for c in prompt_str)
        is_empty = len(prompt_str.strip()) == 0
        log_msg = (
            f"prompt updated: len={len(prompt_str)}, ascii={is_ascii}, "
            f"empty={is_empty}, head={prompt_str[:60]!r}"
        )
        if is_empty:
            logger.warning("prompt validation: 空 prompt を検出 (model が異常動作する可能性)")
        elif not is_ascii:
            logger.warning(f"prompt validation: 非 ASCII prompt を検出 — {log_msg}")
        else:
            logger.info(f"prompt validation: {log_msg}")
        self._last_logged_prompt = prompt_str

    def _apply_head_zero_mask(self, actions: np.ndarray) -> np.ndarray:
        """head 次元 (composite_11d の dim 6, 7) を 0.0 で上書き。

        Issue #187: 学習データの GT は 97%+ が `|val|<0.001`、つまり head は
        ほぼ常に静止。model 出力に乗る小ノイズが実機で意図しない首振りを
        引き起こすため、deploy 時に明示的に 0 マスクする。
        """
        for dim in self._head_dims:
            if actions.ndim == 2:
                if dim < actions.shape[1]:
                    actions[:, dim] = 0.0
            elif actions.ndim == 1:
                if dim < len(actions):
                    actions[dim] = 0.0
        return actions

    def _compute_ensemble(self) -> np.ndarray:
        """指数加重平均を計算"""
        n = len(self._action_buffer)
        if n == 0:
            raise ValueError("アンサンブルバッファが空です")
        if n == 1:
            return self._action_buffer[0].copy()

        w = self._weights[:n]
        w = w / w.sum()
        stacked = np.stack(list(self._action_buffer), axis=0)
        return np.einsum("i,i...->...", w, stacked)

    def _smooth_action(self, actions: np.ndarray) -> np.ndarray:
        """OPT-5: EMA ベースのアクションスムージング。

        隣接アクション間の急激な変化を抑制し、滑らかな動作を実現する。
        Team 18 分析: 滑らかな出力が動作安定性の鍵。
        gripper 等の指定次元は除外（クリスプな遷移が必要なため）。

        alpha=0.3 で jerk -40%, MSE -6.8% の改善を確認 (run24 オフライン検証)。
        """
        first_action = actions[0] if actions.ndim == 2 else actions
        if self._prev_action is not None and len(self._prev_action) == len(first_action):
            alpha = self._action_smoothing_alpha
            smoothed = first_action.copy()
            mask = np.ones(len(first_action), dtype=bool)
            for dim in self._action_smoothing_exclude:
                if dim < len(mask):
                    mask[dim] = False
            smoothed[mask] = alpha * first_action[mask] + (1 - alpha) * self._prev_action[mask]
            if actions.ndim == 2:
                actions[0] = smoothed
            else:
                actions = smoothed
        self._prev_action = (actions[0] if actions.ndim == 2 else actions).copy()
        return actions

    def _clip_gripper(self, actions: np.ndarray) -> np.ndarray:
        """グリッパー次元を [clip_min, clip_max] にクリッピング"""
        idx = self._gripper_idx
        if actions.ndim == 2:
            actions[:, idx] = np.clip(
                actions[:, idx], self._gripper_clip_min, self._gripper_clip_max
            )
        elif actions.ndim == 1:
            actions[idx] = np.clip(
                actions[idx], self._gripper_clip_min, self._gripper_clip_max
            )
        return actions

    def _binarize_gripper(self, actions: np.ndarray) -> np.ndarray:
        """グリッパー次元を閾値で二値化"""
        idx = self._gripper_idx
        if actions.ndim == 2:
            actions[:, idx] = np.where(
                actions[:, idx] >= self._gripper_threshold, 1.0, 0.0
            )
        elif actions.ndim == 1:
            actions[idx] = 1.0 if actions[idx] >= self._gripper_threshold else 0.0
        return actions

    def _apply_force_guard(self, actions: np.ndarray, obs: Dict) -> np.ndarray:
        """力覚センサー閾値ガード

        観測に wrist wrench データが含まれる場合、把持力（fx, fy, fz のノルム）が
        gripper_force_limit を超えていたら、グリッパー指令をこれ以上閉じないよう
        直前の値に固定する。

        対応する観測キー: "wrist_wrench" (6D: [fx, fy, fz, tx, ty, tz])
        キーが存在しない場合は何もしない（フォールバック安全）。
        """
        idx = self._gripper_idx

        # 現在のグリッパー指令値を取得
        if actions.ndim == 2:
            current_gripper = float(actions[0, idx])
        else:
            current_gripper = float(actions[idx])

        # wrench データの取得（なければスキップ）
        wrench = obs.get("wrist_wrench")
        if wrench is not None:
            wrench = np.asarray(wrench, dtype=np.float64)
            force_norm = np.linalg.norm(wrench[:3])  # fx, fy, fz のノルム

            if force_norm > self._gripper_force_limit:
                # 力の上限超過: グリッパーをこれ以上閉じない
                # composite_11d 規約: hand_motor 大=open, 小=close
                # 閉じ方向 = current_gripper < last_gripper_value (値が減少)
                # Issue #197: 比較演算子を < に修正 (旧 > は open 方向ブロック = 反対)
                if self._last_gripper_value is not None and current_gripper < self._last_gripper_value:
                    logger.warning(
                        "力覚ガード発動: force=%.2fN > limit=%.2fN, "
                        "gripper %.3f → %.3f に制限",
                        force_norm, self._gripper_force_limit,
                        current_gripper, self._last_gripper_value,
                    )
                    if actions.ndim == 2:
                        actions[:, idx] = self._last_gripper_value
                    else:
                        actions[idx] = self._last_gripper_value
                    current_gripper = self._last_gripper_value

        self._last_gripper_value = current_gripper
        return actions

    def _apply_contact_guard(self, actions: np.ndarray, obs: Dict) -> np.ndarray:
        """接触検知ガード

        閉じ指令を出しているのにグリッパーの実位置が動かない場合、
        物体に接触していると判定し、これ以上閉じないようブロックする。

        判定ロジック:
          1. 閉じ方向の指令（action[idx] > state[idx]）が出ている
          2. 実位置の変化が tolerance 未満
          3. 上記が patience ステップ連続 → 接触と判定
          4. 接触判定後は閉じ方向をブロック、開く方向は常に許可

        対応する観測キー: "state" (8D+, index=gripper_idx にグリッパー位置)
        キーが存在しない場合は何もしない（フォールバック安全）。
        """
        idx = self._gripper_idx

        # 現在のグリッパー指令値
        if actions.ndim == 2:
            cmd_gripper = float(actions[0, idx])
        else:
            cmd_gripper = float(actions[idx])

        # 観測からグリッパーの実位置を取得
        state = obs.get("state")
        if state is None:
            return actions

        state = np.asarray(state, dtype=np.float64)
        if state.ndim > 1:
            state = state[0]
        if len(state) <= idx:
            return actions

        current_state = float(state[idx])

        # 接触判定のリセット: 開く方向の指令が出たら接触解除
        # composite_11d 規約: 大=open, 小=close → 開き方向 = cmd > last
        # Issue #197: 比較演算子を > に修正 (旧 < は close 方向で誤リセット)
        if self._last_gripper_value is not None and cmd_gripper > self._last_gripper_value:
            if self._contact_detected:
                logger.info("接触検知ガード解除: 開き方向の指令を検出")
            self._contact_detected = False
            self._gripper_stall_count = 0

        # 接触判定中: 閉じ方向をブロック
        # composite_11d 規約: 閉じ方向 = cmd < last
        # Issue #197: 比較演算子を < に修正 (旧 > は open 方向で誤ブロック)
        if self._contact_detected:
            if self._last_gripper_value is not None and cmd_gripper < self._last_gripper_value:
                logger.debug(
                    "接触検知ガード: gripper %.3f → %.3f に制限",
                    cmd_gripper, self._last_gripper_value,
                )
                if actions.ndim == 2:
                    actions[:, idx] = self._last_gripper_value
                else:
                    actions[idx] = self._last_gripper_value
            self._last_gripper_state = current_state
            self._last_gripper_value = float(actions[0, idx]) if actions.ndim == 2 else float(actions[idx])
            return actions

        # ストール検知: 閉じ指令中に位置が動かない
        # composite_11d 規約: 閉じ方向 = cmd < state (cmd は state より小値=閉に向かう)
        # Issue #197: is_closing の定義を修正 (旧 cmd > state は open 方向)
        if self._last_gripper_state is not None and self._last_gripper_value is not None:
            is_closing = cmd_gripper < current_state  # 閉じ方向の指令
            position_change = abs(current_state - self._last_gripper_state)
            is_stalled = position_change < self._gripper_contact_tolerance

            if is_closing and is_stalled:
                self._gripper_stall_count += 1
                if self._gripper_stall_count >= self._gripper_contact_patience:
                    self._contact_detected = True
                    logger.warning(
                        "接触検知ガード発動: %d ステップ連続ストール "
                        "(position_change=%.4f < tolerance=%.4f), "
                        "gripper を %.3f に固定",
                        self._gripper_stall_count,
                        position_change,
                        self._gripper_contact_tolerance,
                        current_state,
                    )
                    # 現在の実位置で固定
                    if actions.ndim == 2:
                        actions[:, idx] = current_state
                    else:
                        actions[idx] = current_state
            else:
                self._gripper_stall_count = 0

        self._last_gripper_state = current_state
        self._last_gripper_value = float(actions[0, idx]) if actions.ndim == 2 else float(actions[idx])
        return actions

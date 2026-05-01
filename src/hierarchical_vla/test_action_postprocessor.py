"""ActionPostprocessor のユニットテスト"""

import numpy as np
import pytest

# テスト用のモック BasePolicy
class MockPolicy:
    def __init__(self, action_output):
        self._action_output = action_output
        self.reset_called = False

    @property
    def metadata(self):
        return {"model": "mock"}

    def infer(self, obs):
        return {"actions": np.array(self._action_output, dtype=np.float64)}

    def reset(self):
        self.reset_called = True


class TestGripperBinarization:
    def test_binarize_2d_above_threshold(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.1, 0.2]])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=True, gripper_threshold=0.5)
        result = pp.infer({})
        assert result["actions"][0, 5] == 1.0

    def test_binarize_2d_below_threshold(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2]])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=True, gripper_threshold=0.5)
        result = pp.infer({})
        assert result["actions"][0, 5] == 0.0

    def test_binarize_at_threshold(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=True, gripper_threshold=0.5)
        result = pp.infer({})
        assert result["actions"][0, 5] == 1.0  # >= threshold → 1.0

    def test_binarize_disabled(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.42, 0.0, 0.0]])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=False)
        result = pp.infer({})
        assert abs(result["actions"][0, 5] - 0.42) < 1e-6

    def test_binarize_does_not_affect_other_dims(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([[0.11, 0.22, 0.33, 0.44, 0.55, 0.8, 0.77, 0.88]])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=True, gripper_threshold=0.5)
        result = pp.infer({})
        a = result["actions"][0]
        assert abs(a[0] - 0.11) < 1e-6
        assert abs(a[4] - 0.55) < 1e-6
        assert abs(a[6] - 0.77) < 1e-6


class TestTemporalEnsemble:
    def test_single_step_no_smoothing(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.8, 7.0, 8.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False, temporal_ensemble=True,
            ensemble_window=3, ensemble_decay=0.5,
        )
        result = pp.infer({})
        np.testing.assert_allclose(result["actions"], actions, atol=1e-6)

    def test_two_steps_ema(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        a1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a2 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock1 = MockPolicy(a1)
        pp = ActionPostprocessor(
            mock1, gripper_binarize=False, temporal_ensemble=True,
            ensemble_window=5, ensemble_decay=0.8,
        )
        pp.infer({})  # step 1: buffer=[a1]

        # step 2
        mock2 = MockPolicy(a2)
        pp._policy = mock2
        result = pp.infer({})  # buffer=[a2, a1]

        # 重み: w0=1.0(最新=a2), w1=0.8(a1) → 正規化 → [1/1.8, 0.8/1.8]
        expected_dim0 = (3.0 * 1.0 + 1.0 * 0.8) / 1.8
        assert abs(result["actions"][0] - expected_dim0) < 1e-6

    def test_buffer_reset(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        a1 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock = MockPolicy(a1)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False, temporal_ensemble=True,
            ensemble_window=3, ensemble_decay=0.5,
        )
        pp.infer({})
        pp.infer({})
        assert len(pp._action_buffer) == 2

        pp.reset_ensemble_buffer()
        assert len(pp._action_buffer) == 0

    def test_full_reset_clears_buffer(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        a = np.array([1.0] * 8)
        mock = MockPolicy(a)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False, temporal_ensemble=True,
        )
        pp.infer({})
        pp.reset()
        assert len(pp._action_buffer) == 0
        assert mock.reset_called


class TestCombined:
    def test_ensemble_then_binarize(self):
        """テンポラルアンサンブル後にグリッパー二値化が適用されることを確認"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # グリッパー値 0.6 → アンサンブル後も ~0.6 → 二値化で 1.0
        a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0])
        mock = MockPolicy(a)
        pp = ActionPostprocessor(
            mock, gripper_binarize=True, gripper_threshold=0.5,
            temporal_ensemble=True, ensemble_window=3, ensemble_decay=0.8,
        )
        result = pp.infer({})
        assert result["actions"][5] == 1.0

    def test_metadata_includes_postprocessor(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        mock = MockPolicy(np.zeros(8))
        pp = ActionPostprocessor(mock, gripper_binarize=True, temporal_ensemble=True)
        meta = pp.metadata
        assert "postprocessor" in meta
        assert meta["postprocessor"]["gripper_binarize"] is True
        assert meta["postprocessor"]["temporal_ensemble"] is True
        assert meta["model"] == "mock"


class TestPACompletionDetection:
    """PA 完了検知の改善テスト（1a, 1b, 1c）"""

    def _make_policy(self, pa_map=None):
        from hierarchical_vla.policy import HierarchicalVLAPolicy

        mock = MockPolicy(np.zeros(8))
        pa_map = pa_map or {"test sht": ["grasp the cup", "place the cup on table"]}
        return HierarchicalVLAPolicy(
            vla_policy=mock,
            pa_decomposition=pa_map,
            config={
                "max_steps": {
                    "navigate": 200,
                    "grasp": 100,
                    "place": 100,
                    "open": 80,
                    "close": 80,
                    "default": 150,
                },
                "min_steps": {
                    "navigate": 5,
                    "grasp": 5,
                    "place": 5,
                    "default": 5,
                },
                "convergence_window": 5,
                "gripper_window": 3,
                "gripper_close_threshold": 0.3,
                "gripper_open_threshold": 0.7,
                "joint_convergence_threshold": 0.01,
                "gripper_grasp_transition_threshold": -0.5,
                "gripper_place_transition_threshold": 0.5,
            },
        )

    # --- 1c: PA 種別ごとの動的 max_steps ---

    def test_max_steps_navigate(self):
        """navigate は 200 ステップまで許容される"""
        from hierarchical_vla.policy import HierarchicalVLAPolicy

        policy = self._make_policy({"test sht": ["navigate to the kitchen"]})
        policy.infer({"prompt": "test sht", "state": np.zeros(8)})
        assert policy._max_steps_for_pa("navigate") == 200

    def test_max_steps_grasp(self):
        """grasp は 100 ステップまで"""
        policy = self._make_policy()
        assert policy._max_steps_for_pa("grasp") == 100

    def test_max_steps_default(self):
        """未知の PA 種別は 150 ステップ"""
        policy = self._make_policy()
        assert policy._max_steps_for_pa("default") == 150

    def test_max_steps_backward_compat(self):
        """max_steps_per_pa（旧設定）が指定された場合のフォールバック"""
        from hierarchical_vla.policy import HierarchicalVLAPolicy

        mock = MockPolicy(np.zeros(8))
        policy = HierarchicalVLAPolicy(
            vla_policy=mock,
            pa_decomposition={"sht": ["grasp cup"]},
            config={"max_steps_per_pa": 50},
        )
        # max_steps_per_pa=50 が上限として機能
        assert policy._max_steps_for_pa("navigate") <= 50

    # --- 1a: スライディングウィンドウ収束判定 ---

    def test_convergence_stable_state(self):
        """関節が安定していれば収束と判定"""
        policy = self._make_policy()
        # 同じ状態を window サイズ分入れる
        stable_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.01, 0.01])
        for _ in range(5):
            policy._state_history.append(stable_state.copy())
        assert policy._is_converged() is True

    def test_convergence_moving_state(self):
        """関節が動いていれば収束しない"""
        policy = self._make_policy()
        for i in range(5):
            state = np.array([0.1 * i, 0.2, 0.3, 0.4, 0.5, 0.5, 0.01, 0.01])
            policy._state_history.append(state)
        assert policy._is_converged() is False

    def test_convergence_insufficient_history(self):
        """履歴が足りなければ収束しない"""
        policy = self._make_policy()
        policy._state_history.append(np.zeros(8))
        assert policy._is_converged() is False

    def test_convergence_with_small_noise(self):
        """微小ノイズは収束と判定（センサーノイズ耐性）"""
        policy = self._make_policy()
        rng = np.random.RandomState(42)
        base_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        for _ in range(10):
            noisy = base_state + rng.normal(0, 0.001, 8)
            policy._state_history.append(noisy)
        assert policy._is_converged() is True

    # --- 1b: グリッパー遷移イベント検知 ---

    def test_gripper_close_transition(self):
        """グリッパーが開→閉に遷移したら検知"""
        policy = self._make_policy()
        # 開状態 (0.8) を window*2 分
        for _ in range(6):
            policy._gripper_history.append(0.8)
        # 閉状態 (0.1) を window 分
        for _ in range(3):
            policy._gripper_history.append(0.1)
        assert policy._detect_gripper_transition("close") is True

    def test_gripper_open_transition(self):
        """グリッパーが閉→開に遷移したら検知"""
        policy = self._make_policy()
        for _ in range(6):
            policy._gripper_history.append(0.1)
        for _ in range(3):
            policy._gripper_history.append(0.9)
        assert policy._detect_gripper_transition("open") is True

    def test_gripper_no_transition(self):
        """グリッパーが動いていなければ遷移なし"""
        policy = self._make_policy()
        for _ in range(10):
            policy._gripper_history.append(0.8)
        assert policy._detect_gripper_transition("close") is False

    def test_gripper_insufficient_history(self):
        """履歴が足りなければ遷移検知しない"""
        policy = self._make_policy()
        policy._gripper_history.append(0.1)
        assert policy._detect_gripper_transition("close") is False

    # --- 統合テスト ---

    def test_navigate_pa_completes_on_convergence(self):
        """navigate PA が関節収束で完了する"""
        policy = self._make_policy({"sht": ["navigate to kitchen", "grasp the cup"]})
        # SHT 開始
        stable_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.01, 0.01])
        policy.infer({"prompt": "sht", "state": stable_state})
        assert policy._current_pa == "navigate to kitchen"

        # min_steps (5) + convergence_window (5) 分の安定状態を投入
        for _ in range(15):
            policy.infer({"prompt": "sht", "state": stable_state})

        # navigate が完了し、次の PA に遷移しているはず
        assert policy._current_pa == "grasp the cup"

    def test_grasp_pa_completes_on_gripper_close(self):
        """grasp PA がグリッパー閉で完了する"""
        policy = self._make_policy({"sht": ["grasp the cup", "place the cup"]})
        # SHT 開始 — グリッパー開状態
        open_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.0, 0.0])
        policy.infer({"prompt": "sht", "state": open_state})
        assert policy._current_pa == "grasp the cup"

        # gripper_window*2 分の開状態
        for _ in range(8):
            policy.infer({"prompt": "sht", "state": open_state})

        # グリッパー閉に遷移
        closed_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.0, 0.0])
        for _ in range(5):
            policy.infer({"prompt": "sht", "state": closed_state})

        assert policy._current_pa == "place the cup"

    def test_forced_transition_at_max_steps(self):
        """max_steps に到達したら強制遷移"""
        policy = self._make_policy({
            "sht": ["grasp the cup", "place the cup"],
        })
        # grasp の max_steps = 100
        moving_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.0, 0.0])
        for i in range(105):
            # 毎回少し状態を変えて収束しないようにする
            s = moving_state.copy()
            s[0] += 0.01 * (i % 10)
            policy.infer({"prompt": "sht", "state": s})

        assert policy._current_pa == "place the cup"

    def test_state_history_cleared_on_pa_advance(self):
        """PA 遷移時に履歴がクリアされる"""
        policy = self._make_policy({"sht": ["navigate to X", "grasp Y"]})
        state = np.zeros(8)
        for _ in range(10):
            policy.infer({"prompt": "sht", "state": state})

        assert len(policy._state_history) > 0
        # 強制的に PA を進める
        policy._pa_queue = ["next pa"]
        policy._advance_pa()
        assert len(policy._state_history) == 0
        assert len(policy._gripper_history) == 0


class TestForceGuard:
    """力覚センサー閾値ガードのテスト"""

    def test_force_guard_blocks_closing_when_force_exceeded(self):
        """力の上限超過時、グリッパーをこれ以上閉じない

        composite_11d 規約: hand_motor 大=open(1.0), 小=close(0.0)
        閉じ方向 = 値が減少する指令 (例: 0.7 → 0.3)
        """
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.1, 0.2])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_force_guard=True, gripper_force_limit=10.0,
        )

        # 1st step: 力が低い → グリッパー 0.7 がそのまま通る
        obs1 = {"wrist_wrench": np.array([1.0, 1.0, 1.0, 0, 0, 0])}
        result1 = pp.infer(obs1)
        assert abs(result1["actions"][5] - 0.7) < 1e-6

        # 2nd step: 力が上限超過 + グリッパーを閉じようとする（0.7 → 0.3、 値が減少）
        actions2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.3, 0.1, 0.2])
        pp._policy = MockPolicy(actions2)
        obs2 = {"wrist_wrench": np.array([8.0, 5.0, 5.0, 0, 0, 0])}  # norm ≈ 10.7
        result2 = pp.infer(obs2)
        # 閉じる方向はブロックされ、前回値 0.7 に制限される
        assert abs(result2["actions"][5] - 0.7) < 1e-6

    def test_force_guard_allows_opening_when_force_exceeded(self):
        """力の上限超過でも、開く方向は許可する

        composite_11d: 開く方向 = 値が増加する指令 (例: 0.3 → 0.7)
        """
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.3, 0.1, 0.2])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_force_guard=True, gripper_force_limit=10.0,
        )

        # 1st step: グリッパー 0.3
        obs1 = {"wrist_wrench": np.array([1.0, 1.0, 1.0, 0, 0, 0])}
        pp.infer(obs1)

        # 2nd step: 力超過だが、グリッパーを開く方向（0.3 → 0.7、 値が増加）→ 許可
        actions2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.1, 0.2])
        pp._policy = MockPolicy(actions2)
        obs2 = {"wrist_wrench": np.array([8.0, 5.0, 5.0, 0, 0, 0])}
        result2 = pp.infer(obs2)
        assert abs(result2["actions"][5] - 0.7) < 1e-6

    def test_force_guard_no_wrench_data(self):
        """wrench データがない場合はスキップ（フォールバック安全）"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.1, 0.2])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_force_guard=True, gripper_force_limit=10.0,
        )
        result = pp.infer({})
        assert abs(result["actions"][5] - 0.8) < 1e-6

    def test_force_guard_below_limit(self):
        """力が上限以下ならグリッパー指令はそのまま通る (閉じ方向も含む)"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.1, 0.2])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_force_guard=True, gripper_force_limit=10.0,
        )

        obs1 = {"wrist_wrench": np.array([1.0, 1.0, 1.0, 0, 0, 0])}
        pp.infer(obs1)

        # 2nd step: 力は上限以下 → 閉じ方向 (0.7 → 0.2) でも許可
        actions2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2])
        pp._policy = MockPolicy(actions2)
        obs2 = {"wrist_wrench": np.array([2.0, 2.0, 2.0, 0, 0, 0])}
        result2 = pp.infer(obs2)
        assert abs(result2["actions"][5] - 0.2) < 1e-6

    def test_force_guard_reset_clears_last_value(self):
        """reset() で last_gripper_value がクリアされる"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_force_guard=True, gripper_force_limit=10.0,
        )
        pp.infer({"wrist_wrench": np.array([1.0, 0, 0, 0, 0, 0])})
        assert pp._last_gripper_value is not None
        pp.reset()
        assert pp._last_gripper_value is None


class TestContactGuard:
    """接触検知ガードのテスト"""

    def _make_pp(self, actions, patience=3, tolerance=0.005):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor
        mock = MockPolicy(actions)
        return ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_contact_guard=True,
            gripper_contact_tolerance=tolerance,
            gripper_contact_patience=patience,
        )

    def test_contact_blocks_closing_after_stall(self):
        """閉じ指令中に位置が動かない場合、patience ステップ後にブロック

        composite_11d 規約: cmd 0.2 < state 0.7 → 閉じ方向 (cmd は state より小値=閉に向かう)
        """
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # 閉じ指令 cmd=0.2 を出し続ける（state=0.7 で動かない=接触）
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2])
        pp = self._make_pp(actions, patience=3)

        obs = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.1, 0.2])}
        # step 1: 初期化（_last_gripper_state セット、比較対象なし）
        pp.infer(obs)
        # step 2-3: stall count 上昇（1, 2）、まだブロックされない
        result2 = pp.infer(obs)
        assert abs(result2["actions"][5] - 0.2) < 1e-6
        result3 = pp.infer(obs)
        assert abs(result3["actions"][5] - 0.2) < 1e-6

        # step 4: patience=3 到達 → 接触検知、実位置 0.7 に固定
        result4 = pp.infer(obs)
        assert abs(result4["actions"][5] - 0.7) < 1e-6

        # step 5: 閉じ方向は引き続きブロック (cmd=0.2 が来ても 0.7 以上に維持)
        result5 = pp.infer(obs)
        assert result5["actions"][5] >= 0.7 - 1e-6

    def test_contact_allows_opening(self):
        """接触検知後も開く方向は許可

        composite_11d 規約: 開く方向 = cmd > last_gripper_value (値が増加)
        """
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions_close = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2])
        pp = self._make_pp(actions_close, patience=2)

        obs = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.1, 0.2])}
        pp.infer(obs)   # step 1: 初期化
        pp.infer(obs)   # step 2: stall=1
        pp.infer(obs)   # step 3: stall=2 → patience=2 到達、接触検知

        # 開く方向の指令（cmd=0.9 > 前回値=0.7）→ 許可 & 接触解除
        actions_open = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.1, 0.2])
        pp._policy = MockPolicy(actions_open)
        result = pp.infer(obs)
        assert abs(result["actions"][5] - 0.9) < 1e-6

    def test_contact_no_state_data(self):
        """state データがない場合はスキップ"""
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2])
        pp = self._make_pp(actions, patience=2)
        result = pp.infer({})
        assert abs(result["actions"][5] - 0.2) < 1e-6

    def test_contact_not_triggered_when_moving(self):
        """位置が動いていれば接触判定されない (close 中に state 減少 = 正常閉動作)"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # cmd=0.2 (closing), state は 0.7 → 0.5 → 0.3 と「閉じている」 = stall でない
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2])
        pp = self._make_pp(actions, patience=2, tolerance=0.005)

        # 位置が毎ステップ動く (close 方向に減少)
        obs1 = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.70, 0.1, 0.2])}
        obs2 = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.50, 0.1, 0.2])}
        obs3 = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.30, 0.1, 0.2])}

        pp.infer(obs1)
        pp.infer(obs2)
        result = pp.infer(obs3)
        # 位置が動いているのでブロックされない
        assert abs(result["actions"][5] - 0.2) < 1e-6

    def test_contact_reset_clears_state(self):
        """reset() で接触状態がクリアされる"""
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.1, 0.2])
        pp = self._make_pp(actions, patience=2)

        obs = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.1, 0.2])}
        pp.infer(obs)   # step 1: 初期化
        pp.infer(obs)   # step 2: stall=1
        pp.infer(obs)   # step 3: stall=2 → 接触検知
        assert pp._contact_detected

        pp.reset()
        assert not pp._contact_detected
        assert pp._gripper_stall_count == 0
        assert pp._last_gripper_state is None


class TestBaseDeadband:
    """Issue #197 B6: base deadband のテスト"""

    def test_base_deadband_zeroes_small_values(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # composite_11d: dim 8=base_x, 9=base_y, 10=base_theta
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.0, 0.0, 0.005, 0.003, 0.008])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            base_deadband=True, base_deadband_threshold=0.01,
        )
        result = pp.infer({})
        a = result["actions"]
        # base 系の |val| < 0.01 は 0 になる
        assert a[8] == 0.0
        assert a[9] == 0.0
        assert a[10] == 0.0
        # arm/gripper は影響なし
        assert abs(a[0] - 0.1) < 1e-6
        assert abs(a[5] - 0.8) < 1e-6

    def test_base_deadband_passes_large_values(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # |val| >= threshold は通過
        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.05, -0.03, 0.02])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            base_deadband=True, base_deadband_threshold=0.01,
        )
        result = pp.infer({})
        a = result["actions"]
        assert abs(a[8] - 0.05) < 1e-6
        assert abs(a[9] - (-0.03)) < 1e-6
        assert abs(a[10] - 0.02) < 1e-6

    def test_base_deadband_disabled_by_default(self):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.005, 0.003, 0.008])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=False)  # base_deadband=False default
        result = pp.infer({})
        a = result["actions"]
        # 何も変化しない
        assert abs(a[8] - 0.005) < 1e-6
        assert abs(a[10] - 0.008) < 1e-6


class TestGripperClosingRateCap:
    """Issue #197 B1': gripper closing rate cap のテスト"""

    def test_closing_rate_cap_limits_close_speed(self):
        """閉じ方向 (cmd < prev) で変化が cap を超えると制限される

        composite_11d: 大=open, 小=close。 cmd 1.0 → 0.5 は close 方向の変化量 0.5。
        cap=0.15 なら 1.0 → 0.85 に制限。
        """
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        mock = MockPolicy(actions1)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_closing_rate_cap=True, gripper_closing_rate_max=0.15,
        )
        # step 1: cmd=1.0 (last_cmd 未設定なのでそのまま)
        r1 = pp.infer({})
        assert abs(r1["actions"][5] - 1.0) < 1e-6

        # step 2: cmd=0.5 (close 方向、 delta=-0.5、 cap=0.15 で制限)
        actions2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        pp._policy = MockPolicy(actions2)
        r2 = pp.infer({})
        assert abs(r2["actions"][5] - 0.85) < 1e-6  # 1.0 - 0.15

    def test_closing_rate_cap_does_not_limit_opening(self):
        """開く方向 (cmd > prev) は cap しない (release は速い方が良い)"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock = MockPolicy(actions1)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_closing_rate_cap=True, gripper_closing_rate_max=0.15,
        )
        pp.infer({})  # step 1: cmd=0.0

        # step 2: cmd=1.0 (open 方向、 delta=+1.0、 cap しない)
        actions2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        pp._policy = MockPolicy(actions2)
        r2 = pp.infer({})
        assert abs(r2["actions"][5] - 1.0) < 1e-6

    def test_closing_rate_cap_small_close_passes(self):
        """変化量 <= cap なら制限なし"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        mock = MockPolicy(actions1)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_closing_rate_cap=True, gripper_closing_rate_max=0.15,
        )
        pp.infer({})

        # close 方向 0.10 → cap 0.15 以下なのでそのまま通る
        actions2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0])
        pp._policy = MockPolicy(actions2)
        r2 = pp.infer({})
        assert abs(r2["actions"][5] - 0.9) < 1e-6

    def test_closing_rate_cap_reset_on_pa_switch(self):
        """reset_pa() で last_gripper_cmd がクリアされる"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            gripper_closing_rate_cap=True, gripper_closing_rate_max=0.15,
        )
        pp.infer({})
        assert pp._last_gripper_cmd is not None

        pp.reset_pa()
        assert pp._last_gripper_cmd is None


class TestPromptValidation:
    """Issue #197 B4: prompt validation logging のテスト"""

    def test_prompt_validation_logs_normal_prompt(self, caplog):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor
        import logging

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False, prompt_validation=True,
        )
        with caplog.at_level(logging.INFO):
            pp.infer({"task": "Pick up the coffee bottle"})
        assert any("prompt validation" in rec.message for rec in caplog.records)

    def test_prompt_validation_warns_on_empty(self, caplog):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor
        import logging

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=False, prompt_validation=True)
        with caplog.at_level(logging.WARNING):
            pp.infer({"task": ""})
        assert any("空 prompt" in rec.message for rec in caplog.records)

    def test_prompt_validation_warns_on_non_ascii(self, caplog):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor
        import logging

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=False, prompt_validation=True)
        with caplog.at_level(logging.WARNING):
            pp.infer({"task": "コーヒーボトルを取る"})
        assert any("非 ASCII" in rec.message for rec in caplog.records)

    def test_prompt_validation_disabled_by_default(self, caplog):
        from hierarchical_vla.action_postprocessor import ActionPostprocessor
        import logging

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=False)  # prompt_validation=False
        with caplog.at_level(logging.INFO):
            pp.infer({"task": "Pick up the coffee"})
        assert not any("prompt validation" in rec.message for rec in caplog.records)


class TestActionClip:
    """Issue #197: action_clip (任意 dim 別 clip、 物理 safety guard) のテスト"""

    def test_action_clip_base_theta_outlier_clipped(self):
        """base_theta (dim 10) の物理限界超過 outlier が clip される

        訓練データに base_theta=1.034 rad/step outlier 確認、 clip ±0.32 で抑制
        """
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # 11D composite: dim 10 = base_theta
        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1, 0.0, 1.034])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            action_clip=True,
            action_clip_ranges={10: [-0.32, 0.32]},
        )
        result = pp.infer({})
        a = result["actions"]
        # base_theta 1.034 → 0.32 に clip
        assert abs(a[10] - 0.32) < 1e-6
        # 他 dim は影響なし
        assert abs(a[5] - 0.5) < 1e-6
        assert abs(a[8] - 0.1) < 1e-6

    def test_action_clip_negative_outlier(self):
        """負の outlier も clip される"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.85])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            action_clip=True,
            action_clip_ranges={10: [-0.32, 0.32]},
        )
        result = pp.infer({})
        assert abs(result["actions"][10] - (-0.32)) < 1e-6

    def test_action_clip_within_range_passes(self):
        """clip 範囲内の値は変更されない"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.15])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            action_clip=True,
            action_clip_ranges={10: [-0.32, 0.32]},
        )
        result = pp.infer({})
        assert abs(result["actions"][10] - 0.15) < 1e-6

    def test_action_clip_multiple_dims(self):
        """複数 dim の同時 clip"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, -0.4, 1.0])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            action_clip=True,
            action_clip_ranges={
                8: [-0.30, 0.30],   # base_x
                9: [-0.30, 0.30],   # base_y
                10: [-0.32, 0.32],  # base_theta
            },
        )
        result = pp.infer({})
        a = result["actions"]
        assert abs(a[8] - 0.30) < 1e-6   # 0.5 → 0.30
        assert abs(a[9] - (-0.30)) < 1e-6  # -0.4 → -0.30
        assert abs(a[10] - 0.32) < 1e-6   # 1.0 → 0.32

    def test_action_clip_disabled_by_default(self):
        """disabled では何も clip されない"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.5])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(mock, gripper_binarize=False)
        result = pp.infer({})
        assert abs(result["actions"][10] - 1.5) < 1e-6

    def test_action_clip_2d_action_chunk(self):
        """2D action chunk でも全 step clip される"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        # (2, 11) chunk
        actions = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.8],
        ])
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            action_clip=True,
            action_clip_ranges={10: [-0.32, 0.32]},
        )
        result = pp.infer({})
        a = result["actions"]
        assert abs(a[0, 10] - 0.32) < 1e-6
        assert abs(a[1, 10] - (-0.32)) < 1e-6

    def test_action_clip_metadata(self):
        """metadata に action_clip / action_clip_ranges が含まれる"""
        from hierarchical_vla.action_postprocessor import ActionPostprocessor

        actions = np.zeros(11)
        mock = MockPolicy(actions)
        pp = ActionPostprocessor(
            mock, gripper_binarize=False,
            action_clip=True,
            action_clip_ranges={10: [-0.32, 0.32]},
        )
        meta = pp.metadata["postprocessor"]
        assert meta["action_clip"] is True
        assert meta["action_clip_ranges"] == {10: [-0.32, 0.32]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

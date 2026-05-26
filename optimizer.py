import numpy as np
import torch


class ArmTrajectoryOptimizer:
    """ロボットアームの軌道・角度指令を探索・最適化するクラス（強化学習・ランダム探索エラー修正版）"""

    def __init__(self, model, num_candidates=30, epsilon=0.10, max_delta=15):
        """
        Args:
            model: models.py で定義した学習済みの EmotionPredictionLSTM
            num_candidates: 1回に探索するランダムな角度候補の数
            epsilon: ランダムな動き（探索）を発生させる確率（0.10 = 10%の確率でランダム）
            max_delta: 通常時の最大角度変化量（度数法。±15度）
        """
        self.model = model
        self.num_candidates = num_candidates
        self.epsilon = epsilon
        self.max_delta = max_delta
        self.model.eval()

    def select_best_angles(self, current_music_token, music_buffer, arm_angles_buffer):
        """
        強化学習風の探索ロジック：
        基本はAIの予測に基づいてベストなハの字角度を選ぶが、
        確率 epsilon (10%) で、新しいデータ収集のために完全ランダムに『きゅっ』と動く。
        """
        # 直前のアームの角度を取得
        last_angles = arm_angles_buffer[-1]

        # 起動直後のゼロ初期化対策（90度フォールバック）
        if np.all(last_angles == 0):
            last_angles = np.array([90.0, 90.0, 90.0, 90.0])

        # ----------------------------------------------------
        # 🎲 確率イプシロン(10%)で「完全なランダム探索」を発動する
        # ----------------------------------------------------
        if np.random.rand() < self.epsilon:
            # AIの予測を無視して、アームが動く限界（45〜135度）の範囲で
            # 現在の角度からちょっと大胆に（最大±25度）ランダムなポーズを強制決定！
            explore_delta = 25
            random_angles = np.zeros(4, dtype=np.int32)
            for idx in range(4):
                current_axis_angle = int(last_angles[idx])
                min_limit = max(45, current_axis_angle - explore_delta)
                max_limit = min(135, current_axis_angle + explore_delta)
                random_angles[idx] = np.random.randint(min_limit, max_limit + 1)
            return random_angles

        # ----------------------------------------------------
        # 🤖 【活用モード】通常のAI予測ベースの最善角度の選別（残り90%の確率）
        # ----------------------------------------------------
        candidate_angles = np.zeros((self.num_candidates, 4), dtype=np.int32)

        for idx in range(4):
            current_axis_angle = int(last_angles[idx])
            min_limit = max(45, current_axis_angle - self.max_delta)
            max_limit = min(135, current_axis_angle + self.max_delta)

            if min_limit >= max_limit:
                candidate_angles[:, idx] = current_axis_angle
            else:
                candidate_angles[:, idx] = np.random.randint(min_limit, max_limit + 1, size=self.num_candidates)

        best_score = -float('inf')
        best_angles = np.array(last_angles, dtype=np.int32)

        with torch.no_grad():
            for angles in candidate_angles:
                next_music_buf = np.roll(music_buffer, -1, axis=0)
                next_music_buf[-1] = current_music_token

                next_arm_buf = np.roll(arm_angles_buffer, -1, axis=0)
                next_arm_buf[-1] = angles

                input_features = np.hstack((next_music_buf, next_arm_buf))
                input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

                predicted_tensor = self.model(input_tensor)
                predicted_emotion = predicted_tensor.squeeze(0).numpy()

                # 💡 【修正箇所】配列インデックス [0] と [1] を明示的に指定して安全に加算
                if len(predicted_emotion) > 1:
                    score = float(predicted_emotion[0] + predicted_emotion[1])
                elif len(predicted_emotion) > 0:
                    score = float(predicted_emotion[0])
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_angles = angles

        return best_angles

import numpy as np
import torch


class ArmTrajectoryOptimizer:
    """ロボットアームの軌道・角度指令を探索・最適化するクラス"""

    def __init__(self, model, num_candidates=50):
        """
        Args:
            model: models.py で定義した学習済みの EmotionPredictionLSTM
            num_candidates: 1回に探索するランダムな角度候補の数（多いほど精密、少ないほど軽量）
        """
        self.model = model
        self.num_candidates = num_candidates
        self.model.eval()

    def select_best_angles(
            self, current_music_token, music_buffer, arm_angles_buffer
    ):
        """
        次のステップに最適な4軸の角度をランダムサンプリングから探索する

        Args:
            current_music_token: 現在のステップの音楽トークン
            music_buffer: 過去の音楽トークン履歴 (history_steps, 1)
            arm_angles_buffer: 過去の角度履歴 (history_steps, 4)
        Returns:
            best_angles: 最も評価の高かった4軸の角度 [a1, a2, a3, a4] (np.array)
        """
        # 1. 未来の角度候補をランダムに大量生成 (例: 50パターン × 4軸)
        # サーボモーターの動く範囲（例: 45度〜135度）に制限
        candidate_angles = np.random.randint(
            45, 136, size=(self.num_candidates, 4)
        )

        best_score = -float("inf")
        best_angles = np.array([90, 90, 90, 90])  # デフォルト値

        # 2. 各候補をシミュレーションして評価
        with torch.no_grad():
            for angles in candidate_angles:
                # 擬似的に次のステップに進めた場合の履歴バッファを作成
                next_music_buf = np.roll(music_buffer, -1, axis=0)
                next_music_buf[-1] = current_music_token

                next_arm_buf = np.roll(arm_angles_buffer, -1, axis=0)
                next_arm_buf[-1] = angles

                # モデルの入力形状 (1, history_steps, 5) に整形
                input_features = np.hstack((next_music_buf, next_arm_buf))
                input_tensor = (
                    torch.tensor(input_features, dtype=torch.float32)
                    .unsqueeze(0)
                )

                # LSTMモデルで未来の表情を予測 (戻り値は [H, S])
                predicted_tensor = self.model(input_tensor)
                predicted_emotion = predicted_tensor.squeeze(0).numpy()

                # 💡 【評価関数】ここでは仮に H と S の合計値が「最も高くなる（良い表情）」と定義
                # 後からここを自由に変更して「特定の感情」を狙い撃ちできます
                score = float(predicted_emotion[0] + predicted_emotion[1])

                # 最高得点の角度を更新
                if score > best_score:
                    best_score = score
                    best_angles = angles

        return best_angles

import numpy as np
import torch

class ArmTrajectoryOptimizer:
    def __init__(self, model, num_candidates=30):
        """
        アームの最高角度を探索するリアルタイムAI最適化クラス。
        """
        self.model = model
        self.num_candidates = num_candidates

    def select_best_angles(self, current_bar_vector, music_buffer, arm_angles_buffer, smoothed_target_angles):
        """
        【安全対策強化版】先読みされた滑らかなベース角度を基準に、
        型エラーを完全に防ぎながらアドリブ候補を選択する。
        """
        # ランダム候補値用のコンテナ配列を用意 (shape: num_candidates, 4)
        candidates = np.zeros((self.num_candidates, 4))

        # 4軸分、先読みで作った「綺麗なベース角度」の周辺に限定してアドリブ（探索）する
        for idx in range(4):
            # 💡【型安全ガード】いかなるオブジェクトが来ても、確実にfloat型として1要素を取り出す
            try:
                base_angle = float(smoothed_target_angles[idx])
            except (TypeError, IndexError, KeyError):
                # 配列でなく単一の関数や変な型が返ってきた場合は安全のためセンター（90度）をベースにする
                base_angle = 90.0

            # 💡【整数キャストガード】計算結果を確実に int型 にして乱数生成器に渡す
            min_limit = int(max(30, int(base_angle) - 12))
            max_limit = int(min(150, int(base_angle) + 12))

            # 逆転または同値になった場合の二重ガード
            if min_limit >= max_limit:
                min_limit = max_limit - 5
                if min_limit < 30:
                    min_limit = 30
                    max_limit = 35

            # 確実に int32 の範囲で候補を作る
            candidates[:, idx] = np.random.randint(
                low=min_limit,
                high=max_limit + 1,
                size=self.num_candidates
            )

        best_score = -float('inf')
        # 初期値も型安全にパース
        try:
            best_angles = np.array([float(x) for x in smoothed_target_angles])
        except Exception:
            best_angles = np.array([90.0, 90.0, 90.0, 90.0])

        # 30個の候補アーム角度それぞれをAIモデルに通して、予測スコアを仮想評価
        for i in range(self.num_candidates):
            candidate_angles = candidates[i]

            temp_music_buffer = music_buffer.copy()
            temp_arm_buffer = arm_angles_buffer.copy()

            temp_music_buffer = np.roll(temp_music_buffer, -1, axis=0)
            temp_music_buffer[-1] = current_bar_vector

            temp_arm_buffer = np.roll(temp_arm_buffer, -1, axis=0)
            temp_arm_buffer[-1] = candidate_angles

            input_features = np.hstack((temp_music_buffer, temp_arm_buffer))
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                predicted_tensor = self.model(input_tensor)
                predicted_emotions = predicted_tensor.squeeze(0).numpy()

            if len(predicted_emotions) > 1:
                score = float(predicted_emotions[0] + predicted_emotions[1])
            else:
                score = float(predicted_emotions)

            if score > best_score:
                best_score = score
                best_angles = candidate_angles.copy()

        return best_angles

import torch
import torch.nn as nn


class EmotionPredictionLSTM(nn.Module):
    """
    【逆モデル】
    入力: 音楽データ (1次元) + ロボットの角度 (4次元) = 5次元
    出力: 人間の予測表情ベクトル (2次元: H, S など)
    """

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, num_layers=2):
        super(EmotionPredictionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 💡 ここを nn.LSTM に修正
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )

        # 全結合レイヤー
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        predicted_emotion = self.fc(last_step_out)
        return predicted_emotion

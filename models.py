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

        # 💡 正しく nn.LSTM に修正済み
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )

        # 全結合レイヤー: LSTMの隠れ状態から人間の表情(H, S)を予測
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        """
        x の形状: (バッチサイズ, 時系列ステップ数, 5)
        """
        out, _ = self.lstm(x)

        # 時系列の「最後のステップ」の出力を取り出して全結合層に渡す
        last_step_out = out[:, -1, :]

        predicted_emotion = self.fc(last_step_out)
        return predicted_emotion


# 💡 循環インポートの原因だった「from models import...」を削除した安全なテストコード
if __name__ == "__main__":
    test_input = torch.randn(1, 32, 5)
    model = EmotionPredictionLSTM()
    output = model(test_input)
    print("モデルの初期化に成功しました！")
    print("テスト入力形状:", test_input.shape)
    print("モデル出力形状 (予測表情):", output.shape)  # (1, 2) になります

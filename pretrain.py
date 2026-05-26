import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import EmotionPredictionLSTM


def generate_pseudo_dance_data(style="ballerina", num_steps=2000, history_steps=32):
    """
    指定されたダンススタイルの『理想の動き（お手本データ）』を自動生成し、
    LSTMの事前学習用データセット (X, Y) に成形する関数
    """
    print(f"🎬 [Pre-train] {style} スタイルの擬似ダンスデータを生成中...")

    # 擬似的な音楽トークン (0〜100のランダム)
    music_tokens = np.random.randint(0, 100, size=(num_steps, 1)).astype(np.float32)

    # 4軸アーム角度の初期化
    arm_angles = np.zeros((num_steps, 4), dtype=np.float32)

    # ダンスの目標値（AIが『この動きが正解だ！』と覚えるためのターゲットスコアデータ）
    # 出力dim=2 に合わせて、[スタイル一致度, 動きの美しさスコア] の2次元として定義
    targets = np.zeros((num_steps, 2), dtype=np.float32)

    for t in range(num_steps):
        if style == "ballerina":
            # 🌸 バレリーナ：左右対称に滑らかなサイン波を描く動き (a1とa3、a2とa4が連動)
            a1 = 90 + np.sin(t * 0.05) * 30
            a2 = 90 + np.cos(t * 0.05) * 25
            a3 = 90 + np.sin(t * 0.05) * 30
            a4 = 90 + np.cos(t * 0.05) * 25

            # この優雅な動きに近いポーズをとっている瞬間の「美しさスコア」を高く設定
            targets[t] = [1.0, 0.9]  # [スタイル適合度, クオリティ]

        elif style == "hiphop":
            # ⚡ HipHop：左右非対称で、一定周期ごとにパキパキと鋭く切り替わる動き
            if (t // 15) % 2 == 0:
                a1, a2, a3, a4 = 120, 60, 60, 120
            else:
                a1, a2, a3, a4 = 60, 120, 120, 60

            targets[t] = [1.0, 0.85]

        else:
            # デフォルト（フォールバック）
            a1, a2, a3, a4 = 90, 90, 90, 90
            targets[t] = [0.0, 0.0]

        arm_angles[t] = [a1, a2, a3, a4]

    # 特徴量の結合 (Token 1次元 + 角度 4次元 = 5次元入力)
    X_all = np.hstack((music_tokens, arm_angles))
    Y_all = targets

    # LSTM用のシーケンス（時系列データ）にスライス
    X_sequences = []
    Y_targets = []
    for i in range(len(X_all) - history_steps):
        X_sequences.append(X_all[i: i + history_steps])
        Y_targets.append(Y_all[i + history_steps])

    return torch.tensor(np.array(X_sequences), dtype=torch.float32), torch.tensor(np.array(Y_targets),
                                                                                  dtype=torch.float32)


def run_pretrain(style="ballerina", epochs=30, model_save_path="best_model.pth"):
    """事前学習を実行してベースとなる重みファイル (best_model.pth) を生成する"""
    print(f"🚀 [Pre-train] {style.upper()} の事前学習を開始します（エポック数: {epochs}）")

    # 5次元入力、2次元出力モデルの初期化
    model = EmotionPredictionLSTM(input_dim=5, hidden_dim=16, output_dim=2)

    # データのロード
    X_train, Y_train = generate_pseudo_dance_data(style=style)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f" └ Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.5f}")

    # 事前学習済みのモデルを保存
    torch.save(model.state_dict(), model_save_path)
    print(f"✨ [Pre-train] 成功！事前学習済みのモデルを '{model_save_path}' に保存しました。")


if __name__ == "__main__":
    # 💡 ここで覚え込ませたいスタイルを指定して単体実行できます
    # "ballerina" または "hiphop"
    run_pretrain(style="ballerina", epochs=30)

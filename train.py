import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from models import EmotionPredictionLSTM


def background_train_loop(csv_path="training_data.csv", model_path="best_model.pth"):
    """
    5秒ごとに直近のCSVデータを監視・読み込み、
    新しく増えたステップのデータ（ミニバッチ）に対して即座にAIモデルを微修正（オンライン学習）するループ。
    """
    print("🤖 [Online Train] 5秒フィードバック型オンライン学習システムが起動しました。")

    # メインスレッドと形状を合わせたモデルを定義 (24次元入力 -> 16隠れ層 -> 2出力)
    model = EmotionPredictionLSTM(input_dim=24, hidden_dim=16, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # 微調整用に少し高めの学習率

    # すでにモデルがある場合はベースとして読み込む
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print("🤖 [Online Train] ベースとなる既存モデルをロードしました。")
        except Exception:
            pass

    last_processed_rows = 0

    while True:
        # ⏰ ぴったり5秒待つ
        time.sleep(5.0)

        if not os.path.exists(csv_path):
            continue

        try:
            # CSVの読み込み（メインスレッドがflushしているため最新値が取れます）
            df = pd.read_csv(csv_path)
            total_rows = len(df)

            # 新しいデータが1ステップ（1行）以上増えていなければスキップ
            if total_rows <= last_processed_rows or total_rows < 32:
                continue

            # 最新の32ステップ分（LSTMの履歴ステップ数分）をバッチとして切り出す
            # 5秒間で増えた直近の成果をAIに叩き込む
            recent_data = df.tail(32).to_numpy()

            # 特徴量のパース
            # current_bar_vector(20列) + display_angles(4列) = 24次元
            X_features = recent_data[:, 1:25].astype(np.float32)
            # ターゲット: 人間のリアルタイム顔ベクトル [face_h, face_s] (最後の2列)
            Y_targets = recent_data[:, 25:27].astype(np.float32)

            # テンソル化 (LSTMの要求形状: batch_size=1, sequence_length=32, input_dim=24)
            X_tensor = torch.tensor(X_features, dtype=torch.float32).unsqueeze(0)
            Y_tensor = torch.tensor(Y_targets[-1], dtype=torch.float32).unsqueeze(0)  # 直近最新の表情を予測ターゲットに

            # ⚡️ 1ステップだけの超高速強化学習（重みの微修正）
            model.train()
            optimizer.zero_grad()
            predictions = model(X_tensor)  # 出力 shape: (1, 2)

            loss = criterion(predictions, Y_tensor)
            loss.backward()
            optimizer.step()

            # 💾 修正された最新のモデルウェイトを即座に上書き保存
            torch.save(model.state_dict(), model_path)

            # メイン側がファイルを検知しやすくするために明示的にタイムスタンプを更新
            os.utime(model_path, None)

            # 進捗の記録
            added_count = total_rows - last_processed_rows
            last_processed_rows = total_rows

            print(
                f"🔥 [Online Train Update] 直近5秒間のデータ({added_count}行)からアームのウケを学習しました！ Loss: {loss.item():.4f}")

        except Exception as e:
            # ファイルの同時アクセスのタイミングバグなどを安全にスルー
            pass

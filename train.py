import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from models import EmotionPredictionLSTM


def background_train_loop(csv_path="music_robot_data.csv", model_save_path="best_model.pth", history_steps=32):
    """
    main.py の裏（バックグラウンドスレッド）で無限に回り続ける学習ループ。
    データが一定数溜まるたびに、自動で最新データを読み込んで再学習を行う。
    """
    print("🤖 [Background Train] バックグラウンド学習スレッドが正常に起動しました。")

    last_row_count = 0
    min_data_required = 100  # 最低限、100ステップ分のデータが溜まるまで待つ

    while True:
        # 1. CSVファイルが存在するか、データが更新されたかをチェック
        if not os.path.exists(csv_path):
            time.sleep(10)  # ファイルができるまで10秒待機
            continue

        try:
            # CSVの読み込み
            df = pd.read_csv(csv_path)
            current_row_count = len(df)
        except Exception:
            time.sleep(2)  # main.py が書き込み中などのコンフリクト回避
            continue

        # 新しいデータが十分に溜まっていない（または増えていない）場合はスキップ
        if current_row_count < min_data_required or current_row_count <= last_row_count:
            time.sleep(10)  # 10秒待機して再チェック
            continue

        print(f"🔄 [Background Train] 新しいデータを検出しました ({current_row_count} 行)。再学習を開始します...")
        last_row_count = current_row_count

        try:
            # 2. データのパースと時系列データセットの作成
            # CSVの構成に応じて、適切なカラム（例: music_token, a1~a4, face_h, face_s）を抽出してください
            # ここでは仮に以下のようにパースしています
            music_tokens = df['music_token'].values.astype(np.float32).reshape(-1, 1)
            arm_angles = df[['a1', 'a2', 'a3', 'a4']].values.astype(np.float32)
            face_emotions = df[['face_h', 'face_s']].values.astype(np.float32)  # 出力正解データ (2次元)

            # 入力特徴量の結合 (Token 1次元 + 角度 4次元 = 5次元)
            X_all = np.hstack((music_tokens, arm_angles))
            Y_all = face_emotions

            # LSTM用のスライディングウィンドウデータ作成
            X_sequences = []
            Y_targets = []
            for i in range(len(X_all) - history_steps):
                X_sequences.append(X_all[i: i + history_steps])
                Y_targets.append(Y_all[i + history_steps])  # 履歴の直後の表情を予測

            if len(X_sequences) == 0:
                time.sleep(5)
                continue

            X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)
            Y_tensor = torch.tensor(np.array(Y_targets), dtype=torch.float32)

            # 3. モデルの初期化と学習処理
            model = EmotionPredictionLSTM(input_dim=5, hidden_dim=16, output_dim=2)

            # すでに過去の学習モデルがあれば重みをロードして引き継ぐ（追加学習）
            if os.path.exists(model_save_path):
                try:
                    model.load_state_dict(torch.load(model_save_path))
                except Exception:
                    pass

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)

            # 簡易的に10エポックだけ回す（バックグラウンドなので軽量に）
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, Y_tensor)
                loss.backward()
                optimizer.step()

            # 4. 学習完了したモデルファイルを上書き保存
            # 一時ファイルに保存してからリネームすることで、main.py 側の読み込みコンフリクトを完全に防ぐ
            tmp_save_path = model_save_path + ".tmp"
            torch.save(model.state_dict(), tmp_save_path)
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            os.rename(tmp_save_path, model_save_path)

            print(f"✨ [Background Train] 学習完了！モデルを更新しました。 最新Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"⚠️ [Background Train] 学習中にエラーが発生しました（データ書き込み中の可能性）: {e}")

        # 次の学習サイクルまで一定時間待機（例: 30秒ごとにデータチェック）
        time.sleep(30)

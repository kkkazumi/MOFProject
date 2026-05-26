import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from models import EmotionPredictionLSTM


def background_train_loop(csv_path="training_data.csv", model_save_path="best_model.pth", history_steps=32):
    """
    main.py の裏で無限に回り続ける学習ループ。
    【エラー可視化＆列名自動解決版】
    """
    print("🤖 [Background Train] バックグラウンド学習スレッドが正常に起動しました。")

    last_row_count = 0
    min_data_required = 100  # 100行以上で開始

    while True:
        if not os.path.exists(csv_path):
            time.sleep(2)
            continue

        try:
            # CSVの行数を軽量に確認
            df = pd.read_csv(csv_path)
            current_row_count = len(df)
        except Exception as e:
            # main.pyが書き込み中のロック競合時は1秒待ってリトライ
            time.sleep(1)
            continue

        # 新しいデータが溜まっていない場合はスキップ
        if current_row_count < min_data_required or current_row_count <= last_row_count:
            time.sleep(2)
            continue

        print(f"🔄 [Background Train] 新しいデータを検出! ({current_row_count} 行)。学習ログの解析中...")
        last_row_count = current_row_count

        try:
            # 💡 【重要】列名に依存せず、位置（インデックス）で安全にデータをパースする
            # 通常、data_mgr.write_row は [bar, music_token, a1, a2, a3, a4, face_h, face_s] の順
            # または末尾に顔ベクトルが入るため、列の並びから自動抽出します。

            # 1. 音楽トークン（通常は2列目、または 'music_token' という名前の列）
            if 'music_token' in df.columns:
                music_tokens = df['music_token'].values.astype(np.float32).reshape(-1, 1)
            else:
                music_tokens = df.iloc[:, 1].values.astype(np.float32).reshape(-1, 1)

            # 2. ロボットの角度4軸（通常は3〜6列目、または 'a1'~'a4'）
            angle_cols = [c for c in df.columns if c in ['a1', 'a2', 'a3', 'a4']]
            if len(angle_cols) == 4:
                arm_angles = df[angle_cols].values.astype(np.float32)
            else:
                arm_angles = df.iloc[:, 2:6].values.astype(np.float32)

            # 3. 人間の顔表情ベクトル（★ここがバグの原因★ 列名に依存せず「最後の2列」を強制取得）
            face_emotions = df.iloc[:, -2:].values.astype(np.float32)

            # 特徴量の結合 (1次元 + 4次元 = 5次元)
            X_all = np.hstack((music_tokens, arm_angles))
            Y_all = face_emotions

            # データ形状チェックの警告
            if X_all.shape[1] != 5 or Y_all.shape[1] != 2:
                print(
                    f"⚠️ [Background Train] データ形状が不正です。X軸:{X_all.shape[1]}軸, Y軸:{Y_all.shape[1]}軸 (5軸と2軸である必要があります)")
                time.sleep(5)
                continue

            # LSTM用の時系列シーケンスを作成
            X_sequences = []
            Y_targets = []
            for i in range(len(X_all) - history_steps):
                X_sequences.append(X_all[i: i + history_steps])
                Y_targets.append(Y_all[i + history_steps])

            if len(X_sequences) == 0:
                time.sleep(2)
                continue

            X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)
            Y_tensor = torch.tensor(np.array(Y_targets), dtype=torch.float32)

            # モデルの初期化と追加学習
            model = EmotionPredictionLSTM(input_dim=5, hidden_dim=16, output_dim=2)
            if os.path.exists(model_save_path):
                try:
                    model.load_state_dict(torch.load(model_save_path))
                except Exception:
                    pass

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)

            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, Y_tensor)
                loss.backward()
                optimizer.step()

            # 一時ファイルを使って安全に保存
            tmp_save_path = model_save_path + ".tmp"
            torch.save(model.state_dict(), tmp_save_path)
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            os.rename(tmp_save_path, model_save_path)

            print(f"✨ [Background Train] 学習完了！モデルファイルを更新しました。 最新Loss: {loss.item():.4f}")

        except Exception as e:
            # 💡 エラー内容を無言で消さず、コンソールに完全に露出させる
            print(f"❌ [Background Train] 学習処理内で致命的エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

        # 次のチェックまで5秒待機
        time.sleep(5)

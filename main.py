import cv2
import numpy as np
import time
import pygame
import torch
import pretty_midi
import threading
import os

from models import EmotionPredictionLSTM
from dataset import MusicDatasetManager
from hardware import HardwareManager
from optimizer import ArmTrajectoryOptimizer
from train import background_train_loop


def main():
    data_mgr = MusicDatasetManager(midi_file="gavotte.mid")
    hw_mgr = HardwareManager(arduino_port="COM3")

    # Pygameオーディオミキサーの初期化
    pygame.mixer.init()
    try:
        full_music_tokens = data_mgr.load_and_vectorize_midi()
    except Exception as e:
        print(e)
        return

    try:
        pygame.mixer.music.load(data_mgr.midi_file)
        print(f"【成功】pygameに {data_mgr.midi_file} を読み込みました。")
    except Exception as e:
        print(f"【エラー】MIDIの再生準備に失敗しました: {e}")
        return

    hw_mgr.init_arduino()
    hw_mgr.init_camera()

    # dataset.py で設定された training_data.csv がここで開かれます
    data_mgr.init_csv_writer()

    # 入力: トークン(1) + 角度(4) = 5次元 / 出力: 表情H, S = 2次元
    model = EmotionPredictionLSTM(input_dim=5, hidden_dim=16, output_dim=2)
    model.eval()

    # 過去の学習データファイル（best_model.pth）が存在していれば最初にロード
    model_path = "best_model.pth"
    last_model_mtime = 0.0
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            last_model_mtime = os.path.getmtime(model_path)
            print("【成功】過去の学習済みモデルをロードしました。")
        except Exception as e:
            print(f"【警告】モデルファイルのロードに失敗しました（初期状態で起動します）: {e}")

    # dataset.pyが使う「training_data.csv」を裏の学習側に直接指定して繋ぐ
    train_thread = threading.Thread(
        target=background_train_loop,
        kwargs={"csv_path": "training_data.csv"},
        daemon=True
    )
    train_thread.start()

    # リアルタイム軌道最適化インスタンスを生成
    trajectory_optimizer = ArmTrajectoryOptimizer(model, num_candidates=30)
    print("【成功】リアルタイムAI最適化システムを初期化しました。")

    # AIモデルの直近履歴ウィンドウバッファ
    history_steps = 32
    music_buffer = np.zeros((history_steps, 1))
    arm_angles_buffer = np.zeros((history_steps, 4))
    predicted_emotions = np.array([0.25, 0.25])

    # 曲本来のBPMと「本物の拍数」を配列エラーなく安全に抽出
    bpm = 120.0
    beats_per_bar = 4.0  # 初期フォールバック値

    try:
        pm = pretty_midi.PrettyMIDI(data_mgr.midi_file)
        tempo_change_times, tempi = pm.get_tempo_changes()
        if len(tempi) > 0:
            bpm = float(tempi[0])
        else:
            bpm = float(pm.estimate_tempo())

        if len(pm.time_signature_changes) > 0:
            beats_per_bar = float(pm.time_signature_changes[0].numerator)
    except Exception as e:
        print(f"【警告】pretty_midiによる解析に失敗しました: {e}")

    try:
        id_to_token_str = {v: k for k, v in data_mgr.tokenizer.vocab.items()}
        for token_id in full_music_tokens:
            if int(token_id) in id_to_token_str:
                token_str = id_to_token_str[int(token_id)]
                if "TimeSig" in token_str:
                    try:
                        sig_part = token_str.split("_")[-1]
                        beats_per_bar = float(sig_part.split("/")[0])
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    # 各種周期の計算
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * beats_per_bar
    min_step_duration = 0.05
    steps_per_bar = int(seconds_per_bar // min_step_duration)
    seconds_per_step = seconds_per_bar / float(steps_per_bar)

    print(
        f"【音楽同期システム起動】本物のBPM: {bpm:.1f} | 自動検出された拍数: {int(beats_per_bar)}拍 | 1小節: {seconds_per_bar:.3f}秒")
    print(f"【ロボット制御最適化】1小節を {steps_per_bar} 分割に決定 ➡️ 制御周期: {seconds_per_step * 1000:.1f} ミリ秒")
    print("\n>>> システムが正常に起動しました！終了は [q] キーです。")

    pygame.mixer.music.play(loops=0)
    start_wall_time = time.time()
    has_started_playing = False

    last_print_bar = -1
    last_record_step = -1
    current_bar_trajectory = np.zeros((steps_per_bar, 4))

    # ビジュアライズ用の現在角度保持変数（初期値は中心の90度）
    display_angles = np.array([90.0, 90.0, 90.0, 90.0])

    while True:
        success, frame = hw_mgr.get_frame()
        if not success:
            break

        if pygame.mixer.music.get_busy():
            has_started_playing = True

        if has_started_playing and not pygame.mixer.music.get_busy():
            print("\n>>> 音楽の再生が完全に終了したため、自動停止します。")
            break

        elapsed_seconds = time.time() - start_wall_time if has_started_playing else 0.0

        current_bar = int(elapsed_seconds / seconds_per_bar) + 1
        total_current_step = int(elapsed_seconds / seconds_per_step)
        step_in_bar = total_current_step % steps_per_bar

        music_index = min(total_current_step, len(full_music_tokens) - 1)
        current_music_token = full_music_tokens[music_index]

        # 表の処理：現在のフレームから顔の数値を解析
        current_face_vector = hw_mgr.analyze_face(frame)

        # 【自動リロード機構】裏で train.py がモデルを更新したかを毎フレーム安全にチェック
        if os.path.exists(model_path):
            try:
                current_mtime = os.path.getmtime(model_path)
                if current_mtime > last_model_mtime:
                    time.sleep(0.05)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    last_model_mtime = current_mtime
                    print("\n🔄 [Main System] 最新の学習済みAIモデルをバックグラウンドから自動リロードしました！")
            except Exception:
                pass

        # 1小節ごとの一括制御情報の送信
        if current_bar != last_print_bar:
            last_print_bar = current_bar

            flat_trajectory_str = ",".join([str(int(90)) for a in range(steps_per_bar * 4)]) + "\n"
            if hw_mgr.ser:
                hw_mgr.ser.write(flat_trajectory_str.encode('utf-8'))

            print(f"⏱️ [{elapsed_seconds:6.2f}s] 🔁 【 第 {current_bar:2d} 小節 制御開始 】 🤖")

        # ステップ毎の、AIモデルによる「最高表情角度」のリアルタイム選別と適用
        if total_current_step != last_record_step:
            last_record_step = total_current_step

            # optimizer.pyを使って次の「ベストな角度4軸」をランダムサンプリング探索
            step_angles = trajectory_optimizer.select_best_angles(
                current_music_token, music_buffer, arm_angles_buffer
            )

            # 画面描画用に角度数値を同期
            display_angles = step_angles

            # バッファの更新
            music_buffer = np.roll(music_buffer, -1, axis=0)
            music_buffer[-1] = current_music_token
            arm_angles_buffer = np.roll(arm_angles_buffer, -1, axis=0)
            arm_angles_buffer[-1] = step_angles

            # 画面表示用の、決定された角度に対する最終予測スコア
            input_features = np.hstack((music_buffer, arm_angles_buffer))
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted_tensor = model(input_tensor)
                predicted_emotions = predicted_tensor.squeeze(0).numpy()

            # 新鮮なデータをCSVへと自動蓄積
            data_mgr.write_row(current_bar, current_music_token, step_angles, current_face_vector)

            # 安全に即座にファイルへ物理保存を行う（フラッシュ処理）
            try:
                for attr_name in dir(data_mgr):
                    attr = getattr(data_mgr, attr_name)
                    if hasattr(attr, 'flush') and not isinstance(attr, str):
                        attr.flush()
            except Exception:
                pass

            # AIがどれくらいの予測スコアで角度を選んだかコンソールに表示
            score = float(predicted_emotions[0] + predicted_emotions[1]) if len(predicted_emotions) > 1 else float(
                predicted_emotions[0])
            print(f"   🎯 [AI選択値] 予測感情スコア(H+S): {score:.4f} ➡️ 決定角度: {step_angles}")

        # 複数要素の配列からインデックスで安全に抽出して描画
        face_h, face_s = 0.0, 0.0
        if current_face_vector is not None:
            try:
                face_h = float(current_face_vector[0])
                if len(current_face_vector) > 1:
                    face_s = float(current_face_vector[1])
            except (TypeError, IndexError, KeyError):
                try:
                    face_h = float(current_face_vector)
                except Exception:
                    pass

        # ----------------------------------------------------
        # 🤖 【変更】サーボ角度をアームの骨組み（棒人間風）としてリアルタイム描画
        # ----------------------------------------------------
        # ----------------------------------------------------
        # 🤖 【ハの字アーム対応】2軸×2本のアーム骨組みをリアルタイム描画
        # ----------------------------------------------------
        h, w, _ = frame.shape

        # 骨（リンク）の長さ（ピクセル単位）
        l1, l2 = 40, 30  # 根元の骨の長さ, 先端の骨の長さ

        # 1. 左アームの描画（画面右下のやや左寄り）
        left_base_x = w - 160
        left_base_y = h - 60

        # 左サーボの角度（a1, a2）を取得
        left_a1 = display_angles[0]
        left_a2 = display_angles[1]

        # ハの字（左下から右上へ傾く構え）を基準とするため、初期角度を135度に設定
        left_ang1 = 135.0 + (left_a1 - 90.0)
        left_rad1 = np.radians(left_ang1)
        left_joint_x = int(left_base_x + l1 * np.cos(left_rad1))
        left_joint_y = int(left_base_y - l1 * np.sin(left_rad1))  # OpenCVは下方向がYプラスのためマイナス

        left_ang2 = left_ang1 + (left_a2 - 90.0)
        left_rad2 = np.radians(left_ang2)
        left_tip_x = int(left_joint_x + l2 * np.cos(left_rad2))
        left_tip_y = int(left_joint_y - l2 * np.sin(left_rad2))

        # 2. 右アームの描画（画面右下のやや右寄り）
        right_base_x = w - 60
        right_base_y = h - 60

        # 右サーボの角度（a3, a4）を取得
        right_a3 = display_angles[2]
        right_a4 = display_angles[3]

        # ハの字（右下から左上へ傾く構え）を基準とするため、初期角度を45度に設定
        right_ang1 = 45.0 - (right_a3 - 90.0)  # 左右対称の動きにするためマイナス反転
        right_rad1 = np.radians(right_ang1)
        right_joint_x = int(right_base_x - l1 * np.cos(right_rad1))
        right_joint_y = int(right_base_y - l1 * np.sin(right_rad1))

        right_ang2 = right_ang1 - (right_a4 - 90.0)
        right_rad2 = np.radians(right_ang2)
        right_tip_x = int(right_joint_x - l2 * np.cos(right_rad2))
        right_tip_y = int(right_joint_y - l2 * np.sin(right_rad2))

        # --- OpenCV画面への描画処理 ---
        # 土台（マウントベース）をグレーの線で結ぶ
        cv2.line(frame, (left_base_x, left_base_y), (right_base_x, right_base_y), (100, 100, 100), 2)
        cv2.circle(frame, (left_base_x, left_base_y), 6, (100, 100, 100), -1)
        cv2.circle(frame, (right_base_x, right_base_y), 6, (100, 100, 100), -1)

        # 左アームの描画（骨：黄緑、関節：赤、先端：オレンジ）
        cv2.line(frame, (left_base_x, left_base_y), (left_joint_x, left_joint_y), (0, 255, 150), 3)
        cv2.line(frame, (left_joint_x, left_joint_y), (left_tip_x, left_tip_y), (0, 255, 150), 3)
        cv2.circle(frame, (left_joint_x, left_joint_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (left_tip_x, left_tip_y), 4, (0, 165, 255), -1)

        # 右アームの描画（左右対称に美しくシンメトリー描画）
        cv2.line(frame, (right_base_x, right_base_y), (right_joint_x, right_joint_y), (0, 255, 150), 3)
        cv2.line(frame, (right_joint_x, right_joint_y), (right_tip_x, right_tip_y), (0, 255, 150), 3)
        cv2.circle(frame, (right_joint_x, right_joint_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (right_tip_x, right_tip_y), 4, (0, 165, 255), -1)

        # 通常の文字テキスト情報描画
        cv2.putText(frame, f"Time: {elapsed_seconds:.2f}s  Bar: {current_bar}  Step: {step_in_bar:02d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Real Face     : H:{face_h:.2f} S:{face_s:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # モデル予測値の安全な取り出し
        pred_h = float(predicted_emotions[0]) if len(predicted_emotions) > 0 else 0.0
        pred_s = float(predicted_emotions[1]) if len(predicted_emotions) > 1 else 0.0
        cv2.putText(frame, f"RuleA Predict : H:{pred_h:.2f} S:{pred_s:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 💡 ここでOpenCVにフレームを表示（1マス戻した正しいインデント位置）
        cv2.imshow('Windows OpenCV AI System', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        # 💡 ここからは while ループの外側の処理（インデントなし）
    pygame.mixer.music.stop()
    data_mgr.close()
    hw_mgr.close()
    print("\n>>> すべてのデータを安全に保存し、終了しました。")


if __name__ == '__main__':
    main()
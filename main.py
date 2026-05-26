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
    data_mgr = MusicDatasetManager(midi_file="furusato.mid")
    hw_mgr = HardwareManager(arduino_port="COM3")

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
    data_mgr.init_csv_writer()

    # 💡 AIモデルの入り口を【24次元（20次元音楽 + 4次元角度）】に変更
    model = EmotionPredictionLSTM(input_dim=24, hidden_dim=16, output_dim=2)
    model.eval()

    model_path = "best_model.pth"
    last_model_mtime = 0.0
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            last_model_mtime = os.path.getmtime(model_path)
            print("【成功】過去の学習済みモデルをロードしました。")
        except Exception as e:
            print(f"【警告】モデルファイルのロードに失敗しました: {e}")

    # 裏の学習側スレッドを起動
    train_thread = threading.Thread(
        target=background_train_loop,
        kwargs={"csv_path": "training_data.csv"},
        daemon=True
    )
    train_thread.start()

    trajectory_optimizer = ArmTrajectoryOptimizer(model, num_candidates=30)
    print("【成功】リアルタイムAI最適化システムを初期化しました。")

    # 💡 履歴バッファの音楽形状を (32, 20) に拡張！
    history_steps = 32
    music_buffer = np.zeros((history_steps, 20))
    arm_angles_buffer = np.zeros((history_steps, 4))
    predicted_emotions = np.array([0.25, 0.25])

    # BPM解析
    bpm = 120.0
    beats_per_bar = 4.0
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
    display_angles = np.array([90.0, 90.0, 90.0, 90.0])
    current_bar_vector = np.zeros(20, dtype=np.float32)

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

        # 💡 リアルタイムに20次元の曲の雰囲気ベクトルを抽出
        current_bar_vector = data_mgr.extract_bar_features(current_bar, seconds_per_bar)

        current_face_vector = hw_mgr.analyze_face(frame)

        # 自動モデルリロード
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

        if current_bar != last_print_bar:
            last_print_bar = current_bar
            flat_trajectory_str = ",".join([str(int(90)) for a in range(steps_per_bar * 4)]) + "\n"
            if hw_mgr.ser:
                hw_mgr.ser.write(flat_trajectory_str.encode('utf-8'))
            print(f"⏱️ [{elapsed_seconds:6.2f}s] 🔁 【 第 {current_bar:2d} 小節 制御開始 】 🤖")

        # ステップ毎の最適化
        if total_current_step != last_record_step:
            last_record_step = total_current_step

            # 💡 20次元の曲の雰囲気ベクトルを渡して最高のハの字角度を探索！
            step_angles = trajectory_optimizer.select_best_angles(
                current_bar_vector, music_buffer, arm_angles_buffer
            )
            display_angles = step_angles

            # 24次元バッファの更新
            music_buffer = np.roll(music_buffer, -1, axis=0)
            music_buffer[-1] = current_bar_vector
            arm_angles_buffer = np.roll(arm_angles_buffer, -1, axis=0)
            arm_angles_buffer[-1] = step_angles

            # 画面表示用のスコア予測
            input_features = np.hstack((music_buffer, arm_angles_buffer))
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted_tensor = model(input_tensor)
                predicted_emotions = predicted_tensor.squeeze(0).numpy()

            # 顔ベクトルのパース
            f_h = float(current_face_vector[0]) if hasattr(current_face_vector, '__len__') and len(
                current_face_vector) > 0 else 0.0
            f_s = float(current_face_vector[1]) if hasattr(current_face_vector, '__len__') and len(
                current_face_vector) > 1 else 0.0

            # 💡 【重要】20次元ベクトルを展開して平らな1行にしてCSVに追記保存！
            data_mgr.write_row_flat(current_bar, current_bar_vector, step_angles, [f_h, f_s])

            # 即座に物理保存（フラッシュ）
            try:
                for attr_name in dir(data_mgr):
                    attr = getattr(data_mgr, attr_name)
                    if hasattr(attr, 'flush') and not isinstance(attr, str):
                        attr.flush()
            except Exception:
                pass

            # AIの判断スコアログ表示
            score = float(predicted_emotions[0] + predicted_emotions[1]) if len(predicted_emotions) > 1 else float(
                predicted_emotions)
            print(f"   🎯 [AI選択値] 予測感情スコア(H+S): {score:.4f} ➡️ 決定角度: {step_angles}")

        # 顔描画用の数値抽出
        face_h = float(current_face_vector[0]) if hasattr(current_face_vector, '__len__') and len(
            current_face_vector) > 0 else 0.0
        face_s = float(current_face_vector[1]) if hasattr(current_face_vector, '__len__') and len(
            current_face_vector) > 1 else 0.0

        # 🤖 【ハの字アーム対応】2軸×2本のアーム骨組みをリアルタイム描画
        h, w, _ = frame.shape
        l1, l2 = 40, 30
        left_base_x, left_base_y = w - 160, h - 60
        left_a1, left_a2 = display_angles[0], display_angles[1]
        left_ang1 = 135.0 + (left_a1 - 90.0)
        left_rad1 = np.radians(left_ang1)
        left_joint_x = int(left_base_x + l1 * np.cos(left_rad1))
        left_joint_y = int(left_base_y - l1 * np.sin(left_rad1))
        left_ang2 = left_ang1 + (left_a2 - 90.0)
        left_rad2 = np.radians(left_ang2)
        left_tip_x = int(left_joint_x + l2 * np.cos(left_rad2))
        left_tip_y = int(left_joint_y - l2 * np.sin(left_rad2))

        right_base_x, right_base_y = w - 60, h - 60
        right_a3, right_a4 = display_angles[2], display_angles[3]
        right_ang1 = 45.0 - (right_a3 - 90.0)
        right_rad1 = np.radians(right_ang1)
        right_joint_x = int(right_base_x - l1 * np.cos(right_rad1))
        right_joint_y = int(right_base_y - l1 * np.sin(right_rad1))
        right_ang2 = right_ang1 - (right_a4 - 90.0)
        right_rad2 = np.radians(right_ang2)
        right_tip_x = int(right_joint_x - l2 * np.cos(right_rad2))
        right_tip_y = int(right_joint_y - l2 * np.sin(right_rad2))

        cv2.line(frame, (left_base_x, left_base_y), (right_base_x, right_base_y), (100, 100, 100), 2)
        cv2.circle(frame, (left_base_x, left_base_y), 6, (100, 100, 100), -1)
        cv2.circle(frame, (right_base_x, right_base_y), 6, (100, 100, 100), -1)
        cv2.line(frame, (left_base_x, left_base_y), (left_joint_x, left_joint_y), (0, 255, 150), 3)
        cv2.line(frame, (left_joint_x, left_joint_y), (left_tip_x, left_tip_y), (0, 255, 150), 3)
        cv2.circle(frame, (left_joint_x, left_joint_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (left_tip_x, left_tip_y), 4, (0, 165, 255), -1)
        cv2.line(frame, (right_base_x, right_base_y), (right_joint_x, right_joint_y), (0, 255, 150), 3)
        cv2.line(frame, (right_joint_x, right_joint_y), (right_tip_x, right_tip_y), (0, 255, 150), 3)
        cv2.circle(frame, (right_joint_x, right_joint_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (right_tip_x, right_tip_y), 4, (0, 165, 255), -1)

        cv2.putText(frame, f"Time: {elapsed_seconds:.2f}s  Bar: {current_bar}  Step: {step_in_bar:02d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Real Face     : H:{face_h:.2f} S:{face_s:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pred_h = float(predicted_emotions[0]) if len(predicted_emotions) > 0 else 0.0
        pred_s = float(predicted_emotions[1]) if len(predicted_emotions) > 1 else 0.0
        cv2.putText(frame, f"RuleA Predict : H:{pred_h:.2f} S:{pred_s:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 画面の表示とキー受付（whileループの内側：スペース8個）
        cv2.imshow('Windows OpenCV AI System', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # プログラムの終了処理（whileループの外側：インデントなし）
    pygame.mixer.music.stop()
    data_mgr.close()
    hw_mgr.close()
    print("\n>>> すべてのデータを安全に保存し、終了しました。")


if __name__ == '__main__':
    main()

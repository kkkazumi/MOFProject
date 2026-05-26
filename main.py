import cv2
import numpy as np
import time
import pygame
import torch
import pretty_midi

from models import RuleALSTM
from dataset import MusicDatasetManager
from hardware import HardwareManager


def generate_bar_trajectory(bar_count, current_time, steps_count, step_duration):
    """
    曲のテンポ・拍数から自動計算された最適なステップ数分、
    かつ指定されたステップ時間（50ms以上）の間隔で未来軌道を生成
    """
    trajectory = np.zeros((steps_count, 4))
    for s in range(steps_count):
        t_offset = s * step_duration
        if bar_count % 2 == 1:
            a1 = 90 + int(np.sin((current_time + t_offset) * 2) * 45)
            a2 = 90 + int(np.cos((current_time + t_offset) * 1.5) * 30)
            a3 = 90 + int(np.sin((current_time + t_offset) * 1) * 45)
            a4 = 90 + int(np.cos((current_time + t_offset) * 0.5) * 30)
        else:
            beat_steps = max(1, steps_count // 4)
            factor = 45 if (s // beat_steps) % 2 == 0 else -45
            a1, a2, a3, a4 = 90 + factor, 90 - factor, 90 + factor, 90 - factor

        trajectory[s] = [a1, a2, a3, a4]
    return trajectory


def main():
    data_mgr = MusicDatasetManager(midi_file="yorokobi.mid")
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
    data_mgr.init_csv_writer()

    model = RuleALSTM(input_dim=5, hidden_dim=16, output_dim=4)
    model.eval()
    print("【成功】ルールA（PyTorch LSTMモデル）を初期化しました。")

    # AIモデルの直近履歴ウィンドウバッファ
    history_steps = 32
    music_buffer = np.zeros((history_steps, 1))
    arm_angles_buffer = np.zeros((history_steps, 4))
    predicted_emotions = np.array([0.25, 0.25, 0.25, 0.25])

    # ----------------------------------------------------
    # 💡 曲本来のBPMと「本物の拍数」を100%自動抽出する（バグ修正版）
    # ----------------------------------------------------
    bpm = 120.0
    beats_per_bar = 4.0  # 初期フォールバック値

    # 1. pretty_midi を使ったファイル解析 ( get_tempo_changes のタプル展開 )
    try:
        pm = pretty_midi.PrettyMIDI(data_mgr.midi_file)

        # 変更時間とテンポ値を別々の配列で受け取る
        tempo_change_times, tempi = pm.get_tempo_changes()
        if len(tempi) > 0:
            bpm = float(tempi[0])  # 曲の最初のBPMを取得
        else:
            bpm = float(pm.estimate_tempo())  # 取得できない場合はノートから推定

        # 拍子（拍数）の取得
        if len(pm.time_signature_changes) > 0:
            beats_per_bar = float(pm.time_signature_changes[0].numerator)
    except Exception as e:
        print(f"【警告】pretty_midiによる解析に失敗しました: {e}")

    # 2. MidiTok のトークン辞書から「TimeSig（拍子）」を逆引きして二重チェック
    try:
        id_to_token_str = {v: k for k, v in data_mgr.tokenizer.vocab.items()}
        for token_id in full_music_tokens:
            if int(token_id) in id_to_token_str:
                token_str = id_to_token_str[int(token_id)]
                if "TimeSig" in token_str:
                    try:
                        sig_part = token_str.split("_")[-1]  # "4/4" など
                        beats_per_bar = float(sig_part.split("/")[0])  # 左側の「4」を取得
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    # 抽出された本物のBPMと本物の拍数から1小節の絶対秒数を動的に計算
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * beats_per_bar

    # 最小ステップ時間を50msに指定
    min_step_duration = 0.05

    # 1小節を50ms以上で等分できる「最適なステップ数」を曲の拍数に合わせて自動計算
    steps_per_bar = int(seconds_per_bar // min_step_duration)
    # 等分された、この曲専用の1ステップあたりの正確な秒数
    seconds_per_step = seconds_per_bar / float(steps_per_bar)

    print(
        f"【音楽同期システム起動】本物のBPM: {bpm:.1f} | 自動検出された拍数: {int(beats_per_bar)}拍 | 1小節: {seconds_per_bar:.3f}秒")
    print(
        f"【ロボット制御最適化】1小節を {steps_per_bar} 分割に決定 ➡️ 制御周期: {seconds_per_step * 1000:.1f} ミリ秒（50ms以上）")
    print("\n>>> システムが正常に起動しました！終了は [q] キーです。")

    pygame.mixer.music.play(loops=0)
    start_wall_time = time.time()
    has_started_playing = False

    last_print_bar = -1
    last_record_step = -1
    current_bar_trajectory = np.zeros((steps_per_bar, 4))

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

        current_face_vector = hw_mgr.analyze_face(frame)

        if current_bar != last_print_bar:
            last_print_bar = current_bar
            current_bar_trajectory = generate_bar_trajectory(
                current_bar, time.time(), steps_count=steps_per_bar, step_duration=seconds_per_step
            )

            flat_trajectory_str = ",".join([str(int(a)) for a in current_bar_trajectory.flatten()]) + "\n"
            if hw_mgr.ser:
                hw_mgr.ser.write(flat_trajectory_str.encode('utf-8'))

            print(
                f"⏱️ [{elapsed_seconds:6.2f}s] 🔁 【 第 {current_bar:2d} 小節 軌道一括送信 】 🤖 ({steps_per_bar}ステップ分を {seconds_per_step * 1000:.1f}ms間隔で一括出力)")

        if total_current_step != last_record_step:
            last_record_step = total_current_step

            safe_step_idx = min(step_in_bar, steps_per_bar - 1)
            step_angles = current_bar_trajectory[safe_step_idx]

            music_buffer = np.roll(music_buffer, -1, axis=0)
            music_buffer[-1] = current_music_token
            arm_angles_buffer = np.roll(arm_angles_buffer, -1, axis=0)
            arm_angles_buffer[-1] = step_angles

            input_features = np.hstack((music_buffer, arm_angles_buffer))
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted_tensor = model(input_tensor)
                predicted_emotions = predicted_tensor.squeeze(0).numpy()

            data_mgr.write_row(current_bar, current_music_token, step_angles, current_face_vector)

        # 配列やリストから個別インデックス指定で安全に値を取得
        try:
            face_h = float(current_face_vector[0])
            face_s = float(current_face_vector[1]) if len(current_face_vector) > 1 else 0.0
        except (TypeError, IndexError):
            face_h = float(current_face_vector) if current_face_vector is not None else 0.0
            face_s = 0.0

        cv2.putText(frame, f"Time: {elapsed_seconds:.2f}s  Bar: {current_bar}  Step: {step_in_bar:02d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.putText(frame, f"Real Face     : H:{face_h:.2f} S:{face_s:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"RuleA Predict : H:{float(predicted_emotions[0]):.2f} S:{float(predicted_emotions[1]):.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Windows OpenCV AI System', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    pygame.mixer.music.stop()
    data_mgr.close()
    hw_mgr.close()
    print("\n>>> すべてのデータを安全に保存し、終了しました。")


if __name__ == '__main__':
    main()

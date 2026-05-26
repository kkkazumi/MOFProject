import csv
import os
import numpy as np
import pretty_midi


class MusicDatasetManager:
    def __init__(self, midi_file="gavotte.mid"):
        self.midi_file = midi_file
        self.csv_file = "training_data.csv"
        self.csv_f = None
        self.writer = None

        # MidiTokのダミー初期化（互換性維持のため残す）
        class DummyTokenizer:
            def __init__(self):
                self.vocab = {"TimeSig_4/4": 1, "TimeSig_3/4": 2}

        self.tokenizer = DummyTokenizer()

    def load_and_vectorize_midi(self):
        """互換性維持のためのダミートークン配列を返す"""
        return np.array([1, 2, 1, 2])

    def init_csv_writer(self):
        """💡 【追記・20次元拡張版】ファイルがなければ新次元ヘッダーを作成、あれば追記モードで開く"""
        file_exists = os.path.exists(self.csv_file)

        # 'w' から 'a' (追記モード) に変更
        self.csv_f = open(self.csv_file, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_f)

        # 新しくファイルを作った最初の1回目だけ、20次元に完全対応したヘッダーを書き込む
        if not file_exists:
            # 12次元のクロマ（コード）列名
            chroma_headers = [f"chroma_{i}" for i in range(12)]
            # 8次元のリズム列名
            rhythm_headers = [f"rhythm_{i}" for i in range(8)]
            # すべて合体
            headers = ["bar_numb"] + chroma_headers + rhythm_headers + ["a1", "a2", "a3", "a4", "face_h", "face_s"]
            self.writer.writerow(headers)

    def write_row_flat(self, bar, music_vector, arm_angles, face_vector):
        """💡 【新設】20次元の音楽ベクトル、角度4軸、顔2軸を1本の平らな行にして確実にCSVへ追記する"""
        if self.writer:
            # 各要素をリストとしてフラットに結合
            row_data = [bar] + list(music_vector) + list(arm_angles) + list(face_vector)
            self.writer.writerow(row_data)

    def extract_bar_features(self, current_bar, seconds_per_bar):
        """💡 【1小節特徴量抽出】コード12次元 ＋ リズム8次元 ＝ 計20次元の曲の雰囲気ベクトルを生成"""
        chroma_vector = np.zeros(12, dtype=np.float32)
        rhythm_vector = np.zeros(8, dtype=np.float32)

        try:
            bar_start_time = (current_bar - 1) * seconds_per_bar
            bar_end_time = current_bar * seconds_per_bar

            pm = pretty_midi.PrettyMIDI(self.midi_file)

            # --- ① コード（和音）情報の抽出（12次元） ---
            chroma = pm.get_chroma(fs=10)
            times = np.linspace(0, pm.get_end_time(), chroma.shape[1])
            bar_indices = np.where((times >= bar_start_time) & (times <= bar_end_time))[0]
            if len(bar_indices) > 0:
                chroma_vector = np.mean(chroma[:, bar_indices], axis=1).astype(np.float32)
                if np.max(chroma_vector) > 0:
                    chroma_vector = chroma_vector / np.max(chroma_vector)

            # --- ② 時系列（リズム）情報の抽出（8次元） ---
            step_duration = seconds_per_bar / 8.0
            for instrument in pm.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    if bar_start_time <= note.start < bar_end_time:
                        relative_time = note.start - bar_start_time
                        grid_idx = int(relative_time // step_duration)
                        if 0 <= grid_idx < 8:
                            rhythm_vector[grid_idx] = 1.0

        except Exception as e:
            pass

        return np.concatenate([chroma_vector, rhythm_vector])

    def close(self):
        if self.csv_f:
            self.csv_f.close()

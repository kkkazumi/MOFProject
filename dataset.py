import os
import csv
from miditok import REMI, TokenizerConfig


class MusicDatasetManager:
    def __init__(self, midi_file="yorokobi.mid", csv_file="training_data.csv", vocab_file="midi_vocab.txt"):
        self.midi_file = midi_file
        self.csv_file = csv_file
        self.vocab_file = vocab_file
        self.full_music_tokens = []
        self.csv_f = None
        self.csv_writer = None

        config = TokenizerConfig(num_velocities=16, use_chords=True, use_tempos=True)
        self.tokenizer = REMI(config)

    def load_and_vectorize_midi(self):
        if not os.path.exists(self.midi_file):
            raise FileNotFoundError(f"{self.midi_file} が見つかりません。")

        midi_tokens = self.tokenizer(self.midi_file)
        self.full_music_tokens = []
        self._extract_ids(midi_tokens)

        with open(self.vocab_file, "w", encoding="utf-8") as vf:
            vf.write("=== MidiTok トークンID ➡️ 音楽情報 翻訳辞書 ===\n")
            sorted_vocab = sorted(self.tokenizer.vocab.items(), key=lambda x: x[1])
            for token_str, token_id in sorted_vocab:
                vf.write(f"ID: {token_id:3d}  ->  Meaning: {token_str}\n")
        return self.full_music_tokens

    def _extract_ids(self, obj):
        if hasattr(obj, 'ids'):
            self._extract_ids(obj.ids)
        elif isinstance(obj, list):
            if len(obj) > 0 and isinstance(obj[0], list):
                self._extract_ids(obj[0])
            else:
                for item in obj:
                    self._extract_ids(item)
        else:
            try:
                self.full_music_tokens.append(float(obj))
            except (TypeError, ValueError):
                pass

    def init_csv_writer(self):
        self.csv_f = open(self.csv_file, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_f)
        self.csv_writer.writerow([
            "bar_number", "music_token",
            "arm_angle_1", "arm_angle_2", "arm_angle_3", "arm_angle_4",
            "face_happy", "face_sad", "face_angry", "face_surprise"
        ])

    def write_row(self, bar, token, single_step_angles, face_vector):
        if self.csv_writer:
            self.csv_writer.writerow([
                bar, int(token),
                int(single_step_angles[0]), int(single_step_angles[1]),
                int(single_step_angles[2]), int(single_step_angles[3]),
                f"{float(face_vector[0]):.2f}", f"{float(face_vector[1]):.2f}",
                f"{float(face_vector[2]):.2f}", f"{float(face_vector[3]):.2f}"
            ])

    def close(self):
        if self.csv_f:
            self.csv_f.close()

    def extract_bar_features(self, current_bar, seconds_per_bar):
        """
        指定された小節の『リズム情報（8分音符刻み: 8次元）』と『コード情報（12音階: 12次元）』を
        時系列や和音の文脈を壊さずに計20次元のベクトルとして抽出する。
        """
        import pretty_midi
        import numpy as np

        # 1. 音楽のコード情報（12次元クロマベクトル）の初期化
        # [C, C#, D, D#, E, F, F#, G, G#, A, A#, B] の12音階の強さ
        chroma_vector = np.zeros(12, dtype=np.float32)

        # 2. 音楽のリズム情報（8分音符刻みの発音フラグ: 8次元）の初期化
        # 4分音符が4つの4拍子（1小節）を8等分して、音が新しく鳴ったタイミングを 1 / 0 で記録
        rhythm_vector = np.zeros(8, dtype=np.float32)

        try:
            # 現在の小節の開始時間と終了時間を秒数で計算
            # current_bar は 1 から始まるため、時間を 0 スタートに補正
            bar_start_time = (current_bar - 1) * seconds_per_bar
            bar_end_time = current_bar * seconds_per_bar

            # pretty_midiオブジェクトを再ロード（または事前に保持しているものを使用）
            pm = pretty_midi.PrettyMIDI(self.midi_file)

            # --- ① コード（和音）情報の抽出 ---
            # 指定した小節の区間だけを対象に、鳴っている音の成分（クロマベクトル）を取得
            # get_chromaのサンプリング周波数を指定し、時間区間でスライス
            chroma = pm.get_chroma(fs=10)
            times = np.linspace(0, pm.get_end_time(), chroma.shape[1])

            # 現在の小節区間に入っているサンプルのインデックスを取得
            bar_indices = np.where((times >= bar_start_time) & (times <= bar_end_time))[0]
            if len(bar_indices) > 0:
                # 区間内の音の強さを平均して12次元ベクトルにする
                chroma_vector = np.mean(chroma[:, bar_indices], axis=1).astype(np.float32)
                # 最大値が 1.0 になるように正規化（AIが学習しやすくするため）
                if np.max(chroma_vector) > 0:
                    chroma_vector = chroma_vector / np.max(chroma_vector)

            # --- ② 時系列（リズム）情報の抽出 ---
            step_duration = seconds_per_bar / 8.0  # 1小節を8等分（8分音符の長さ）

            for instrument in pm.instruments:
                if instrument.is_drum:
                    continue  # ドラム以外（メロディや伴奏）の発音タイミングを重視

                for note in instrument.notes:
                    # 音が鳴り始めたタイミング（Note On）がこの小節内にあるかチェック
                    if bar_start_time <= note.start < bar_end_time:
                        # 小節内のどこ（0〜7番目のグリッド）で鳴ったかを計算
                        relative_time = note.start - bar_start_time
                        grid_idx = int(relative_time // step_duration)
                        if 0 <= grid_idx < 8:
                            rhythm_vector[grid_idx] = 1.0  # 発音ありフラグを立てる

        except Exception as e:
            print(f"⚠️ [MusicDatasetManager] 1小節ベクトルの抽出に失敗しました: {e}")

        # 12次元（コード）と 8次元（リズム）をドッキングさせて20次元ベクトルとして返す
        return np.concatenate([chroma_vector, rhythm_vector])

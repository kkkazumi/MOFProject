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
n
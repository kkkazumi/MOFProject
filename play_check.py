# play_check.py
import time
import os
import pygame

midi_file = "d-kirakira.mid"

if not os.path.exists(midi_file):
    print(f"【エラー】{midi_file} が見つかりません。プロジェクトフォルダーにあるか確認してください。")
    exit()

print(">>> スピーカーの音量を少し上げてお待ちください...")

# Pythonの音声機能を初期化してロード
pygame.mixer.init()
pygame.mixer.music.load(midi_file)

# 1回だけ再生
pygame.mixer.music.play(loops=0)
print("♪♪♪ いま twinkle.mid をPythonから直接再生しています ♪♪♪")

# 音が鳴っている間（最大10秒間）プログラムを維持する
start_time = time.time()
while pygame.mixer.music.get_busy() and (time.time() - start_time) < 50:
    time.sleep(0.1)

pygame.mixer.music.stop()
print(">>> 再生が終了しました。")

# create_midi.py
import cv2 # OpenCVの内部にあるMidiファイル保存機能を利用（追加インストール不要）
import numpy as np

# MidiTokのテスト用に、簡単な「きらきら星（ド・ド・ソ・ソ）」のMIDIを生成する簡易スクリプト
# ※もし動かない場合は手持ちのMIDIファイルを "twinkle.mid" にリネームして配置してもOKです
import os
print("実験用の twinkle.mid を生成中...")
# シンプルなMIDIバイナリ（きらきら星の冒頭）を直接書き出します
midi_data = b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0MTrk\x00\x00\x004\x00\x90<\x60\x81\x00\x80<\x00\x00\x90<\x60\x81\x00\x80<\x00\x00\x90G\x60\x81\x00\x80G\x00\x00\x90G\x60\x81\x00\x80G\x00\x00\xff\x2f\x00'
with open("twinkle.mid", "wb") as f:
    f.write(midi_data)
print("twinkle.mid の生成が完了しました！")

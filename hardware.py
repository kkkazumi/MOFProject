import cv2
import os
import time
import serial
import numpy as np


class HardwareManager:
    def __init__(self, arduino_port="COM3"):
        self.arduino_port = arduino_port
        self.ser = None
        self.cap = None
        self.face_cascade = None
        self.eye_cascade = None

    def init_arduino(self):
        try:
            self.ser = serial.Serial(self.arduino_port, 9600, timeout=0.1)
            time.sleep(2)
            print(f"【成功】Arduino ({self.arduino_port}) に接続しました。")
        except Exception:
            print(f"【警告】Arduinoが見つかりません。ダミーモードで実行します。")
            self.ser = None

    def init_camera(self):
        cv2_dir = os.path.dirname(cv2.__file__)
        face_xml = os.path.join(cv2_dir, 'data', 'haarcascade_frontalface_default.xml')
        eye_xml = os.path.join(cv2_dir, 'data', 'haarcascade_eye.xml')

        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Webカメラが起動できません。")

    def get_frame(self):
        if self.cap:
            return self.cap.read()
        return False, None

    def analyze_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        current_face_vector = np.array([0.25, 0.25, 0.25, 0.25])  # 初期値

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            if len(eyes) >= 2:
                current_face_vector = np.array([0.50, 0.10, 0.10, 0.30])
            elif len(eyes) == 1:
                current_face_vector = np.array([0.20, 0.50, 0.20, 0.10])
            else:
                current_face_vector = np.array([0.10, 0.20, 0.60, 0.10])
            break
        return current_face_vector

    def send_angles(self, angles):
        angle_str_list = [str(int(a)) for a in angles]
        angle_string = ",".join(angle_str_list) + "\n"
        if self.ser:
            self.ser.write(angle_string.encode('utf-8'))
        return angle_str_list

    def close(self):
        if self.cap:
            self.cap.release()
        if self.ser:
            self.ser.close()
        cv2.destroyAllWindows()

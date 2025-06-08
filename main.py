import cv2, sys, os
import numpy as np
from scipy import signal
from collections import deque
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

CASCADE_PATH = resource_path("data/haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

class PulseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pulse Monitor")
        self.root.geometry("1200x800")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.ax1 = self.figure.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, padx=10, pady=10)

        self.signal_data = deque()
        self.capture_duration = 15
        self.fps = 60
        self.max_frames = int(self.capture_duration * self.fps)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = True
        self.frame_count = 0

        self.root.after(1, self.update_frame)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        display_text = ""
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            forehead_y = y + int(0.15 * h)
            forehead_h = int(0.15 * h)
            forehead = frame[forehead_y:forehead_y+forehead_h, x+int(w*0.25):x+int(w*0.75)]

            hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin = cv2.bitwise_and(forehead, forehead, mask=mask)

            avg_intensity = np.mean(skin)
            self.signal_data.append(avg_intensity)

            cv2.rectangle(frame, (x+int(w*0.25), forehead_y), (x+int(w*0.75), forehead_y+forehead_h), (0,255,0), 2)
            display_text = "STAY STILL"

            text_x = 10 if x + w//2 > frame.shape[1] // 2 else frame.shape[1] - 200
            text_y = frame.shape[0] - 30 if y < frame.shape[0] // 2 else 30
            cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        self.frame_count += 1
        if self.frame_count >= self.max_frames:
            self.running = False
            self.cap.release()
            self.process_signal()
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        max_w = self.root.winfo_width() // 2
        max_h = self.root.winfo_height()
        scale = min(max_w / w, max_h / h, 1.0)
        resized_frame = cv2.resize(rgb_frame, (int(w * scale), int(h * scale)))

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def process_signal(self):
        raw_signal = np.array(self.signal_data)
        smoothed_signal = np.convolve(raw_signal, np.ones(5)/5, mode='valid')

        nyquist = 0.5 * self.fps
        low_cutoff, high_cutoff = 0.8 / nyquist, 2.5 / nyquist
        b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='bandpass')
        filtered_signal = signal.filtfilt(b, a, smoothed_signal)

        peaks, _ = signal.find_peaks(filtered_signal, distance=int(self.fps/2), prominence=0.5)
        bpm = len(peaks) * 60 / (len(filtered_signal) / self.fps)

        self.ax1.clear()
        self.ax1.plot(filtered_signal, label='Filtered')
        self.ax1.plot(peaks, filtered_signal[peaks], "rx")
        self.ax1.set_title(f"Filtered Signal — BPM: {bpm:.2f}")

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PulseApp(root)
    root.mainloop()

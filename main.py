import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import requests
from PIL import Image
from io import BytesIO

SERVER_URL = "http://localhost:8000/query"  # Your backend endpoint

class RealSenseViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Robotics Frontend")
        self.resize(1200, 600)

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # RealSense Video Label
        self.video_label = QLabel("Camera Stream")
        left_layout.addWidget(self.video_label)

        # Query input
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type query, e.g. 'apple'")
        self.submit_button = QPushButton("Submit Query")
        self.submit_button.clicked.connect(self.submit_query)
        left_layout.addWidget(self.input_line)
        left_layout.addWidget(self.submit_button)

        # Masked Image Label
        self.mask_label = QLabel("Masked Image")
        right_layout.addWidget(self.mask_label)

        # Combine layouts
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Timer for updating video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Store coordinates
        self.last_coordinates = None

    def update_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def submit_query(self):
        query = self.input_line.text().strip()
        if not query:
            return
        # Capture current frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        frame = np.asanyarray(color_frame.get_data())
        # Encode frame as JPEG for sending
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {"query": query}
        try:
            response = requests.post(SERVER_URL, data=data, files=files)
            response.raise_for_status()
            resp_json = response.json()
            # Display masked image
            mask_bytes = requests.get(resp_json["masked_image_url"]).content
            pil_img = Image.open(BytesIO(mask_bytes))
            qt_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, QImage.Format_RGB888)
            self.mask_label.setPixmap(QPixmap.fromImage(qt_img))
            # Get coordinates
            coords = resp_json["coordinates"]  # e.g., [x, y, z]
            self.last_coordinates = coords
            # Send to robot script
            self.send_to_robot(coords)
        except Exception as e:
            print(f"Error: {e}")

    def send_to_robot(self, coords):
        # This is an example, adapt as needed.
        # For IPC, you can use sockets, files, or a message queue.
        with open("coords.txt", "w") as f:
            f.write(str(coords))

    def closeEvent(self, event):
        self.pipeline.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = RealSenseViewer()
    viewer.show()
    sys.exit(app.exec_())

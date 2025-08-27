from ultralytics import YOLO
import numpy as np
import cv2
import torch
import mss
import time

model = YOLO("/Users/lucbaumeler/Documents/Eth/VsCode/MLS/RL/JassKing-3.0/swisslos_vision/best.pt")  # or yolo11s.pt, etc.
sct = mss.mss()
monitor = sct.monitors[1]

last_time = 0

while True:
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    results = model(frame)
    annotated_frame = results[0].plot()

    current_time = time.time()
    if current_time - last_time > 1:
        last_time = current_time
        classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        print(f"Detected classes: {classes}")

    cv2.imshow("YOLOv11 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
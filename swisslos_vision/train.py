from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")  # or yolo11s.pt, etc.
model.train(
    data="/Users/lucbaumeler/Documents/Eth/VsCode/MLS/RL/JassKing-3.0/swisslos_vision/data/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu"  # GPU 0, or "cpu" if no GPU
)
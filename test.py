import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import torch

device = 'cuda'

model = YOLO("runs/detect/train16/weights/best.pt")
path ="result.mp4"


# This shows the result....
results = model.predict(source=path, show = True)  


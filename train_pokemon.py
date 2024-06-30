from ultralytics import YOLO

# building new model from scratch
model = YOLO("yolov8n.pt")

results = model.train(data="./Pokemon Detection.v2i.yolov8/data.yaml",epochs=40,imgsz=640)


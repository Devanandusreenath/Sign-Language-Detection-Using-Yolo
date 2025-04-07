from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
results = model.train(data="input1/data.yaml", epochs=250, imgsz=640)  # train the model

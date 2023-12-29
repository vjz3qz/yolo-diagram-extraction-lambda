from ultralytics import YOLO

# load model
model = YOLO('yolov8n.pt') # replace with your custom model pt file

# export model
model.export(format="onnx", dynamic=True)
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt") 
# Use the model
model.train(data="/opt/ml/baseline/baseline_cv11/yolov8/ultralytics/datasets/recycle.yaml",epochs=100)  # train the model
# metrics=model.val() # validset으로 모델 평가
# results=model("https://ultralyics.com/images/bus.jpg") #predict on the image
# success=model.export(format="onnx") # ONNX 포맷으로 모델 export

from ultralytics import YOLO
import cv2

# 1. Load a pre-trained YOLOv8 nano model (lightweight for lab use)
model = YOLO('yolov8n.pt')

# 2. Run inference on an image
# Replace 'bus.jpg' with your image path or a URL
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True, conf=0.5)

# 3. Process results
for r in results:
    print(f"Detected {len(r.boxes)} objects.")
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class ID: {cls}, Confidence: {conf:.2f}")

# The annotated image is saved in 'runs/detect/predict'
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision.transforms import functional as F

# 1. Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Load and Preprocess Image
img = Image.open("your_image.jpg").convert("RGB")
img_tensor = F.to_tensor(img).unsqueeze(0) # Add batch dimension

# 3. Prediction
with torch.no_grad():
    prediction = model(img_tensor)

# 4. Output results (Bounding boxes, Labels, Scores)
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Filter for high confidence detections
threshold = 0.8
for i in range(len(scores)):
    if scores[i] > threshold:
        print(f"Label: {labels[i]}, Score: {scores[i]:.2f}, Box: {boxes[i].tolist()}")
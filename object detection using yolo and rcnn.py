import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

# --- PART 1: YOLOv8 (One-Stage Detection) ---
print("--- Running YOLOv8 ---")
yolo_model = YOLO('yolov8n.pt')

# Run YOLO and save the automatic annotation
# Note: This saves to 'runs/detect/predict/'
yolo_results = yolo_model.predict(source='https://www.stripeuk.com/wp-content/uploads/2018/11/shutterstock_540175210-BW.jpg', save=True, conf=0.5)

print(f"YOLOv8: Detected {len(yolo_results[0].boxes)} objects.")
print(f"YOLO image saved in: {yolo_results[0].save_dir}")


# --- PART 2: Faster R-CNN (Two-Stage Detection) ---
print("\n--- Running Faster R-CNN ---")
# Load modern weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn_model = fasterrcnn_resnet50_fpn(weights=weights)
rcnn_model.eval()

# Load the image downloaded by YOLO (bus.jpg)
img_path = 'car.jpeg'
img = Image.open(img_path).convert("RGB")
img_tensor = F.to_tensor(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    prediction = rcnn_model(img_tensor)

# --- PART 3: Visualization for Faster R-CNN ---
def save_rcnn_results(image, prediction, threshold=0.8):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    count = 0
    for i in range(len(scores)):
        if scores[i] > threshold:
            count += 1
            box = boxes[i].detach().cpu().numpy()
            # Draw Rectangle: (x, y), width, height
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            # Label with Score
            plt.text(box[0], box[1], f"ID:{labels[i]} {scores[i]:.2f}",
                     color='white', fontsize=8, bbox={'facecolor': 'green', 'alpha': 0.5})

    plt.title(f"Faster R-CNN Detections (Threshold > {threshold})")
    plt.axis('off')

    # Save the manual R-CNN annotation
    output_name = 'faster_rcnn_output.png'
    plt.savefig(output_name, bbox_inches='tight')
    print(f"Faster R-CNN: Detected {count} high-confidence objects.")
    print(f"R-CNN image saved as: {output_name}")
    plt.show()

save_rcnn_results(img, prediction)
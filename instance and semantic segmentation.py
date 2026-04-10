import torch
import torchvision.transforms.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import os

# 1. Download a sample image if it doesn't exist
IMAGE_PATH = 'sample_image.jpg'
if not os.path.exists(IMAGE_PATH):
    print("Downloading sample image...")
    # Using a standard street scene image which is great for segmentation
    url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"
    urllib.request.urlretrieve(url, IMAGE_PATH)

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

def perform_semantic_segmentation(image_path):
    print("--- Running Semantic Segmentation (DeepLabV3) ---")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    preprocess = weights.transforms()
    input_batch = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    output_predictions = output.argmax(0).byte().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(output_predictions, cmap='nipy_spectral')
    ax2.set_title("Semantic Segmentation (DeepLabV3)")
    ax2.axis('off')
    plt.show()

def perform_instance_segmentation(image_path, threshold=0.75):
    print("--- Running Instance Segmentation (Mask R-CNN) ---")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    # Move predictions back to CPU for matplotlib visualization
    masks = prediction['masks'][prediction['scores'] > threshold].squeeze(1)
    masks = masks.cpu().numpy()

    img_np = np.array(img)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)

    colors = np.random.uniform(0, 255, size=(len(masks), 3))

    for i, mask in enumerate(masks):
        color_mask = np.zeros_like(img_np, dtype=np.uint8)
        color_mask[mask > 0.5] = colors[i]
        plt.imshow(color_mask, alpha=0.5)

    plt.title(f"Instance Segmentation - {len(masks)} objects detected")
    plt.axis('off')
    plt.show()

# Execute the models
perform_semantic_segmentation(IMAGE_PATH)
perform_instance_segmentation(IMAGE_PATH, threshold=0.7)
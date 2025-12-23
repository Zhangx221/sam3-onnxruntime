import numpy as np
import cv2
from PIL import Image
import requests

def preprocess_image(image):
    """Preprocesses the image for the model."""
    image = cv2.resize(image, (1008, 1008), interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])

    # 从 HWC → CHW
    image = np.transpose(image, (2, 0, 1))
    # 增加 batch 维度 → (1, C, H, W)
    image = np.expand_dims(image, axis=0)

    return image


if __name__ == "__main__":
    image_url = "onnx_export/zidane.jpg"

    image = cv2.imread(image_url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = preprocess_image(image)
    print(processed_image.shape)
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from PIL import Image
import numpy as np
import os

# Paths
MODEL_DIR = "/Users/abhinavrai/Downloads/rt-detr"

class RTDETRInference:
    def __init__(self, model_dir=MODEL_DIR):
        self.processor = RTDetrImageProcessor.from_pretrained(model_dir)
        self.model = RTDetrForObjectDetection.from_pretrained(model_dir)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
        else:
            self.model.to('cpu')

    def infer(self, image: np.ndarray, threshold: float = 0.3):
        """
        image: numpy array in HWC, BGR or RGB
        threshold: confidence threshold for filtering results
        Returns: list of dicts with boxes, scores, labels
        """
        if image.shape[2] == 3:
            pil_img = Image.fromarray(image[..., ::-1])  # BGR to RGB
        else:
            pil_img = Image.fromarray(image)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[pil_img.size[::-1]]
        )[0]
        # results: dict with boxes, scores, labels
        return results

# Example usage:
if __name__ == "__main__":
    import cv2
    img = cv2.imread("/path/to/your/image.jpg")
    inferencer = RTDETRInference()
    results = inferencer.infer(img)
    print(results)

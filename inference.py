import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    # Read face detection configuration; use defaults if not provided
    face_config = config.get("face_detect", {})
    input_dir = Path(face_config.get("input_dir", "./data/celeba_raw"))
    output_dir = Path(face_config.get("output_dir", "./cropped_faces"))
    max_images = face_config.get("max_images", None)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Get list of image files (filtering by common image extensions)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if max_images is not None:
        all_images = all_images[:int(max_images)]

    print(f"Processing {len(all_images)} images from {input_dir}")

    for img_path in all_images:
        # Read image using OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Unable to read {img_path}")
            continue

        # Convert BGR (OpenCV) to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces = detector.detect_faces(img_rgb)

        if faces:
            # Get bounding box of the first face detected
            x, y, width, height = faces[0]['box']
            # Ensure positive coordinates
            x, y = max(x, 0), max(y, 0)
            x2, y2 = x + width, y + height

            # Crop the face region
            cropped_face = img_rgb[y:y2, x:x2]

            # Save the cropped face
            face_pil = Image.fromarray(cropped_face)
            save_path = output_dir / img_path.name
            face_pil.save(str(save_path))
        else:
            print(f"No face detected in {img_path}")

    print("Face cropping completed!")

if __name__ == "__main__":
    main()

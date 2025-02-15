import torch
from facenet_pytorch import MTCNN
from PIL import Image
import random
from pathlib import Path
import time

def detect_faces(config):
    """PyTorch-based face detection with configurable parameters"""
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    cfg = config['face_detection']
    
    # Initialize MTCNN detector with configurable parameters
    detector = MTCNN(
        keep_all=False,  # Detect only the largest face
        min_face_size=cfg['min_face_size'],
        thresholds=cfg['detection_thresholds'],
        post_process=False,
        device=device
    )
    
    # Create output directory
    output_dir = Path(cfg['cropped_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_dir = Path(cfg['orig_dir'])
    image_paths = [p for p in input_dir.glob('*.*') 
                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    
    processed = 0
    start_time = time.time()
    
    for img_path in image_paths[:cfg['max_faces']]:
        try:
            # Load image with PIL
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            
            # Detect faces
            boxes, _ = detector.detect(image)
            
            if boxes is not None:
                # Select largest face
                box = boxes[0]
                x, y, x2, y2 = box
                w = x2 - x
                h = y2 - y
                
                # Apply random jitter
                jitter = int(cfg['jitter_factor'] * min(w, h))
                x += random.randint(-jitter, jitter)
                y += random.randint(-jitter, jitter)
                w += random.randint(-jitter, jitter)
                h += random.randint(-jitter, jitter)
                
                # Ensure valid coordinates
                x = max(0, x)
                y = max(0, y)
                w = min(w, original_size[0] - x)
                h = min(h, original_size[1] - y)
                
                if w < cfg['min_face_size'] or h < cfg['min_face_size']:
                    continue
                
                # Crop and save
                cropped = image.crop((x, y, x + w, y + h))
                cropped.save(output_dir / img_path.name)
                processed += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"✅ Cropped {processed} faces to {output_dir} in {elapsed:.2f} seconds")
    print(f"Processing speed: {processed/elapsed:.2f} images/sec")
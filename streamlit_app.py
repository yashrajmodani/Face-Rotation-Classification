# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from model import get_model
import yaml
from pathlib import Path
import torchvision.transforms as transforms

# Load configuration
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize device
device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')

# Load model with caching
@st.cache_resource
def load_trained_model():
    model = get_model(config['model_name'], config['num_classes'])
    model_path = Path(config['model_saving']['save_dir']) / \
                config['model_name'] / "exp1" / "best" / "best_resnet34_epoch039_valacc84.26_20250207-160217.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

model = load_trained_model()

# Initialize face detector
mtcnn = MTCNN(
    keep_all=False,
    min_face_size=config['face_detection']['min_face_size'],
    thresholds=config['face_detection']['detection_thresholds'],
    device=device
)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_and_preprocess(image):
    """Detect face, crop, and preprocess image"""
    # Detect face
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None
    
    # Get largest face
    x, y, x2, y2 = boxes[0]
    cropped = image.crop((x, y, x2, y2))
    
    # Apply transformations
    return transform(cropped).unsqueeze(0).to(device)

def get_prediction(tensor):
    """Get class probabilities from model"""
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def main():
    st.title("🎭 Face Orientation Classification")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a face image", 
                                   type=['jpg', 'jpeg', 'png'],
                                   help="Upload a clear frontal face image")
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add predict button
            if st.button("Predict Orientation"):
                # Process image
                processed_tensor = detect_and_preprocess(image)
                
                if processed_tensor is not None:
                    # Get prediction
                    probs = get_prediction(processed_tensor)
                    class_names = ['North (315-45°)', 'West (225-315°)', 
                                 'South (135-225°)','East (45-135°)' ]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Create columns for layout
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Predicted Class", 
                                f"{class_names[np.argmax(probs)]}",
                                help="The detected face orientation class")
                        st.metric("Confidence", 
                                f"{np.max(probs)*100:.1f}%",
                                help="Model's confidence in the prediction")
                    
                    with col2:
                        st.subheader("Class Probabilities")
                        for name, prob in zip(class_names, probs):
                            st.progress(
                                float(prob),
                                text=f"{name}: {prob*100:.1f}%"
                            )
                else:
                    st.error("Not classified in any class")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
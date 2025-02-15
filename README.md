Below is an updated version of the README with an added section for dataset links and project evaluation documents. Simply copy and paste the text (including the triple backticks) into your `README.md` file:

```markdown
# Face Rotation Classifier 🚀

Welcome to the **Face Rotation Classifier** project! This repository implements a system to detect and classify the rotation angle of faces in images. It supports data augmentation (both on-the-fly and preprocessed), model training, evaluation, inference, and even face detection with cropping. All key settings are managed through a single configuration file (`config.yaml`), making it super flexible and easy to use.

---

## 📋 Table of Contents

- [Features ✨](#features-)
- [Project Structure 📂](#project-structure-)
- [Installation and Setup 💻](#installation-and-setup-)
- [Configuration ⚙️](#configuration-)
- [Usage 🚀](#usage-)
  - [Preprocessing 🛠️](#preprocessing-)
  - [Training and Testing 🔥](#training-and-testing-)
  - [Inference 🔍](#inference-)
  - [Face Detection 🤖](#face-detection-)
  - [Visualization 🎨](#visualization-)
  - [Streamlit Web App 🌐](#streamlit-web-app-)
- [Dataset & Project Documentation 📑](#dataset--project-documentation-)
- [Troubleshooting ⚠️](#troubleshooting-)
- [License 📄](#license-)

---

## Features ✨

- **Flexible Data Augmentation:**  
  Choose between on-the-fly augmentation and preprocessed augmentation.

- **Multiple Model Options:**  
  Use models like ResNet18, ResNet34, ResNet50, ResNet101, VGG16, MobileNetV2, Inception, ViT, AlexNet, or a custom basic model.

- **Centralized Configuration:**  
  All hyperparameters and file paths are managed in `config.yaml`.

- **Face Detection:**  
  Utilizes MTCNN for detecting and cropping faces.

- **Web Interface:**  
  A simple Streamlit app to upload an image and view its predicted rotation.

- **Visualization:**  
  Easily visualize augmented images in a grid to inspect augmentation quality.

---

## Project Structure 📂

```
├── config.yaml                  - Central configuration file

├── data_loader.py               - Dataset and preprocessing functions

├── model.py                     - Model definitions and utility function to load models

├── preprocess.py                - Preprocessing script for image augmentation

├── train.py                     - Training loop and validation logic

├── test.py                      - Evaluation script for testing the model

├── inference.py                 - Inference script for single image prediction

├── visualize.py                 - Visualization script for augmented images
  
├── face_detect.py               - Face detection and cropping using MTCNN

├── streamlit_app.py             - Streamlit web interface for model inference

└── main.py                      - Main entry point for preprocessing, training/testing, or inference


## Installation and Setup 💻

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/face-rotation-classifier.git
   cd face-rotation-classifier
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**

   All required packages are listed in `requirements.txt`. Install them using:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If you plan to use CUDA, make sure you have a compatible version of PyTorch installed.

---

## Configuration ⚙️

All settings are managed via `config.yaml`. This file includes:

- **Augmentation Parameters:**  
  Blur radius, color jitter, crop scale, noise intensity, etc.

- **Model Settings:**  
  Choose the model (e.g., `"resnet18"`), number of classes, batch size, learning rate, epochs, and image size.

- **System Settings:**  
  Number of workers, CUDA usage, and process type (preprocessing, training+testing, or testing only).

- **Data Directories:**  
  Paths for raw data, preprocessed data, training, and testing.

- **Face Detection Settings:**  
  Input and output directories for face detection, plus a maximum image limit.

Simply modify `config.yaml` to match your environment and preferences.

---

## Usage 🚀

### Preprocessing 🛠️

To generate augmented images and organize them into class-specific folders:

1. **Set `process_type` to `1`** in `config.yaml`.
2. **Run the main script:**

   ```bash
   python main.py --config config.yaml
   ```

### Training 🔥

To train the model:

1. **Set `process_type` to `2`** in `config.yaml`.
2. **Run the main script:**

   ```bash
   python main.py --config config.yaml
   ```

   Training will save the best model (based on validation accuracy) in a designated folder under `saved_models/`.

### Testing 🔥

To test the model:

1. **Set `process_type` to `3`** in `config.yaml`.
2. **Run the main script:**

   ```bash
   python main.py --config config.yaml
   ```

   Testing will evaluate the best model (based on validation accuracy) in a designated folder under `saved_models/`.

### Inference 🔍

To predict the rotation angle of a single image:

1. Ensure you have a trained model and update the model path if necessary.
2. **Run the inference script:**

   ```bash
   python inference.py test_image.jpg
   ```

   The script will display the image along with its predicted rotation (e.g., 0°, 90°, etc.).

### Face Detection 🤖

To run face detection and crop faces from images:

1. Configure the `face_detect` section in `config.yaml` (set the input directory, output directory, and optionally `max_images`).
2. **Run the face detection script:**

   ```bash
   python face_detect.py
   ```

   Cropped face images will be saved in the specified output directory.

### Visualization 🎨

To visualize a grid of augmented images (default is a 5x5 grid):

1. **Set `process_type` to `4`** in `config.yaml`.
2. **Run the main script:**

   ```bash
   python main.py --config config.yaml
   ```

   Will save a image of 5 * 5 grid of transformed images with name augmentations.jpg.

### Streamlit Web App 🌐

To launch the web interface for image prediction:

1. **Start the Streamlit app:**

   ```bash
   streamlit run streamlit_app.py
   ```

2. Open the URL provided by Streamlit in your browser, upload an image, and view its predicted rotation.

---

## Dataset & Project Documentation 📑

- **Dataset Links:**  
  - The Training dataset used for this project can be downloaded from: [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
  - Alternatively, access testing version at: [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces)

- **Project Evaluation Documents:**  
  - **Project Report (PDF):** Detailed project evaluation and methodology can be found in the [Project_Report.pdf](docs/Project_Report.pdf) file.
  - **Project Presentation (PPT):** A presentation summarizing the project is available as [Project_Presentation.ppt](docs/Project_Presentation.pptx).

Feel free to refer to these documents for an in-depth understanding of the project and its evaluation.

---

## Troubleshooting ⚠️

- **Palette Image Warnings / Conversion Issues:**  
  If you see warnings regarding palette images with transparency, the dataset loader now converts these images properly. If issues persist, verify that your dataset images are not corrupted.

- **Best Model Not Found:**  
  If training was halted mid-epoch and experiment info is missing, the testing script will search for the best available model in the experiment folder. You can adjust the `load_best_model_path` function in `test.py` if needed.

- **CUDA Issues:**  
  Ensure you have the correct version of PyTorch and that CUDA is available if you're using GPU acceleration.

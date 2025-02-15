import yaml
import argparse
from pathlib import Path

def load_config(config_name):
    with open(config_name) as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Rotation Classifier')
    parser.add_argument('--config', help='Path to configuration file')   
    args = parser.parse_args()
    config_name = args.config
    config = load_config(config_name)
    
    if config['process_type'] == 1:
        from preprocess import preprocess_data
        preprocess_data(config)
    
    elif config['process_type'] == 2:
        from train import train_model
        train_model(config)
    
    elif config['process_type'] == 3:
        from test import test_model
        test_model(config)

    elif config['process_type'] == 4:
        from face_detect import detect_faces
        detect_faces(config)

    elif config['process_type'] == 5:
        from visualize import visualize_augmentations
        visualize_augmentations(config)
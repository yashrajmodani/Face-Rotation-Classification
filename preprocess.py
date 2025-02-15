import yaml
from pathlib import Path
from data_loader import UnifiedDataset  # For config validation

def preprocess_data(config):
    from data_loader import DatasetPreprocessor  # Lazy import
    
    print("\n🚀 Starting preprocessing...")
    Path(config['reg_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['orig_dir']).mkdir(parents=True, exist_ok=True)
    
    preprocessor = DatasetPreprocessor(config)
    preprocessor.preprocess_dataset()
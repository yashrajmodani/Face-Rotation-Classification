import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
from model import get_model
from data_loader import get_loaders

def load_best_model_path(config):
    # Determine architecture folder: saved_models/{model_name}
    arch = config['model_name']
    base_dir = Path(config['model_saving']['save_dir']) / arch
    # Get experiment folders (e.g. exp1, exp2, …) and select the one with the highest number
    exp_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("exp")]
    if not exp_folders:
        raise ValueError("No experiment folders found for architecture " + arch)
    exp_folders = sorted(exp_folders, key=lambda d: int(d.name.replace("exp", "")))
    latest_exp = exp_folders[-1]
    # Load experiment_info.json from the latest experiment folder
    exp_info_path = latest_exp / "experiment_info.json"
    if not exp_info_path.exists():
        raise ValueError("Experiment info not found in " + str(latest_exp))
    with open(exp_info_path, "r") as f:
        exp_info = json.load(f)
    best_model_path = exp_info.get("best_model", {}).get("path", None)
    if best_model_path is None:
        raise ValueError("No best model found in experiment info.")
    return best_model_path

def test_model(config):
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    
    # Load test data
    if config['augmentation_mode'] == 'preprocessed':
        _, test_set = get_loaders(config)
    else:
        _, test_set = get_loaders(config)
    
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], 
                             num_workers=config['num_workers'], pin_memory=True)
    
    # Determine best model path from the latest experiment for this architecture
    best_model_path = load_best_model_path(config)
    print("Loading best model from:", best_model_path)
    
    model = get_model(config['model_name'], config['num_classes']).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Evaluate model performance
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import classification_report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['0°', '90°', '180°', '270°']))

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    test_model(config)
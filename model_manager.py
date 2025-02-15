
import torch
import json
from pathlib import Path
from datetime import datetime
import re

class ModelManager:
    def __init__(self, config):
        # Save the full configuration for keys like model_name
        self.full_config = config
        self.config = config['model_saving']
        # Base save directory (e.g. "./saved_models")
        self.base_save_dir = Path(self.config['save_dir'])
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get architecture name from configuration (e.g. "resnet34")
        arch = self.full_config.get('model_name', 'model')
        # Create top-level folder for this architecture: saved_models/resnet34
        self.arch_dir = self.base_save_dir / arch
        self.arch_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine experiment number by counting existing "exp" folders under this architecture
        exp_dirs = [d for d in self.arch_dir.iterdir() if d.is_dir() and re.match(r"exp\d+", d.name)]
        self.exp_num = len(exp_dirs) + 1
        # Create new experiment folder inside the architecture folder
        self.exp_dir = self.arch_dir / f"exp{self.exp_num}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for checkpoints and best model inside the experiment folder
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.exp_dir / "best"
        self.best_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_filename(self, metrics, epoch, mode):
        """Generate a standardized filename using the config format."""
        fmt_params = {
            'model_name': self.full_config.get('model_name', 'model'),
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
            **{k: metrics.get(k, 0.0) for k in self.config.get('metrics_to_include', [])}
        }
        filename = self.config['filename_format'].format(**fmt_params)
        # For best mode, prefix with "best_"
        if mode == "best":
            filename = "best_" + filename
        filename = filename.replace(' ', '_').replace(':', '-') + f".{self.config['archive_format']}"
        return filename

    def save_model(self, model, metrics, epoch, mode="checkpoint"):
        """Save model weights and metadata into the appropriate folder.
           mode can be "checkpoint" (for every 5th/last epoch) or "best".
        """
        if mode == "checkpoint":
            save_folder = self.checkpoints_dir
        elif mode == "best":
            save_folder = self.best_dir
            # Remove any existing best model file so that only one best model exists.
            for file in save_folder.glob("*"):
                file.unlink()
        else:
            raise ValueError("Unsupported mode. Use 'checkpoint' or 'best'.")
            
        filename = self._generate_filename(metrics, epoch, mode)
        save_path = save_folder / filename
        
        # Save model weights
        torch.save(model.state_dict(), save_path)
        
        # Save metadata if enabled
        if self.config.get('save_metadata', False):
            metadata = {
                'saved_at': datetime.now().isoformat(),
                'epoch': epoch,
                'config': {k: v for k, v in self.config.items() if k != 'metadata_fields'},
                'metrics': metrics,
                'model_architecture': str(model)
            }
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        print(f"[{mode.upper()}] Model saved at: {save_path}")
        return save_path

    def save_experiment_info(self, info):
        """Save experiment summary information in the experiment folder."""
        info_path = self.exp_dir / "experiment_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Experiment info saved at: {info_path}")

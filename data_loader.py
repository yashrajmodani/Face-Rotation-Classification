import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import shutil

class UnifiedDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.transform = self._get_transforms()
        
        if config['augmentation_mode'] == 'preprocessed':
            self.data_dir = Path(config['reg_dir'])
            self.image_paths = list(self.data_dir.rglob('*.*'))
        else:
            # Use train_dir for training; test_dir for testing.
            self.data_dir = Path(config['train_dir' if mode == 'train' else 'test_dir'])
            self.image_paths = [f for f in self.data_dir.glob('*.*') 
                               if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        # Check that the dataset is not empty.
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in directory: {self.data_dir}. "
                             "Ensure the directory exists and contains images with extensions .jpg, .jpeg, or .png.")

        import random  # add this if not already imported elsewhere
        if self.mode == 'train' and config.get('max_train_images'):
            self.image_paths = random.sample(self.image_paths, min(len(self.image_paths), config['max_train_images']))
        elif self.mode == 'test' and config.get('max_test_images'):
            self.image_paths = random.sample(self.image_paths, min(len(self.image_paths), config['max_test_images']))
        print(f"Using {len(self.image_paths)} images for {self.mode}.")
        
    def _get_transforms(self):
        if self.config['augmentation_mode'] == 'preprocessed':
            return transforms.Compose([
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.config['image_size'], scale=(0.8, 1.0)),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.config['augmentation_mode'] == 'preprocessed':
            label = int(img_path.parent.name)
        else:
            angle = np.random.uniform(0, 360)
            if 315 <= angle or angle < 45: label = 0
            elif 45 <= angle < 135: label = 1
            elif 135 <= angle < 225: label = 2
            else: label = 3
            image = image.rotate(angle, expand=True)
        
        return self.transform(image), label

def get_loaders(config):
    if config['augmentation_mode'] == 'preprocessed':
        full_dataset = UnifiedDataset(config, 'train')
        return random_split(full_dataset, [int(0.8*len(full_dataset)), len(full_dataset)-int(0.8*len(full_dataset))])
    else:
        train_set = UnifiedDataset(config, 'train')
        test_set = UnifiedDataset(config, 'test')
        return train_set, test_set
    

class DatasetPreprocessor:
    def __init__(self, config):
        self.config = config
        self.orig_dir = Path(config['orig_dir'])
        self.reg_dir = Path(config['reg_dir'])
        self.angle_ranges = [(0, 45), (45, 135), (135, 225), (225, 315)]

    def _clear_reg_dir(self):
        if self.reg_dir.exists():
            shutil.rmtree(self.reg_dir)
        self.reg_dir.mkdir(parents=True)

    def preprocess_dataset(self):
        from tqdm import tqdm
        
        self._clear_reg_dir()
        source_images = [f for f in self.orig_dir.glob('*.*') 
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')][:self.config['max_source_images']]
        
        for class_idx in range(4):
            (self.reg_dir / str(class_idx)).mkdir(parents=True, exist_ok=True)

        print("Generating augmented dataset...")
        for img_path in tqdm(source_images):
            image = Image.open(img_path).convert('RGB')
            for _ in range(self.config['versions_per_image']):
                angle = np.random.uniform(0, 360)
                rotated = image.rotate(angle, expand=True)
                
                # Determine class label
                if 315 <= angle or angle < 45: label = 0
                elif 45 <= angle < 135: label = 1
                elif 135 <= angle < 225: label = 2
                else: label = 3
                
                save_dir = self.reg_dir / str(label)
                save_path = save_dir / f"{img_path.stem}_rot{angle:.1f}{img_path.suffix}"
                rotated.save(save_path)
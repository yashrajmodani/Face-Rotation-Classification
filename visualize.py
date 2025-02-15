import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader import UnifiedDataset

def visualize_augmentations(config, mode='train', save_path='augmentations.png'):
    """Visualize 25 random augmented images in a 5x5 grid"""
    # Create dataset and loader
    dataset = UnifiedDataset(config, mode)
    loader = DataLoader(dataset, batch_size=25, shuffle=True, num_workers=config['num_workers'])
    
    # Get a batch of augmented images
    images, labels = next(iter(loader))
    
    # Reverse normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denormalized = images * std + mean
    
    # Create plot
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        img = denormalized[i].permute(1, 2, 0).numpy()
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved augmentation visualization to {save_path}")
    plt.show()

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    visualize_augmentations(config)

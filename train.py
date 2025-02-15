import torch
from torch.utils.data import DataLoader
from model import get_model
from data_loader import get_loaders

def train_model(config):
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device}")
    
    # Load training and validation datasets
    if config['augmentation_mode'] == 'preprocessed':
        train_set, val_set = get_loaders(config)
    else:
        train_set, val_set = get_loaders(config)
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'],
                            num_workers=config['num_workers'], pin_memory=True)
    
    # Setup model, optimizer, and loss function
    model = get_model(config['model_name'], config['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize ModelManager for experiment saving
    from model_manager import ModelManager
    model_manager = ModelManager(config)
    
    best_acc = 0.0
    best_model_path = None
    experiment_info = {
        "experiment_number": model_manager.exp_num,
        "model_name": config['model_name'],
        "epochs": config['epochs'],
        "learning_rate": config['learning_rate'],
        "batch_size": config['batch_size'],
        "saved_checkpoints": [],
        "best_model": {}
    }
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        # Evaluate on validation set
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        metrics = {"val_acc": val_acc, "train_loss": avg_train_loss}
        
        # Save checkpoint only if this is every 5th epoch or the final epoch
        if (epoch+1) % 5 == 0 or (epoch+1) == config['epochs']:
            checkpoint_path = model_manager.save_model(model, metrics, epoch=epoch+1, mode="checkpoint")
            experiment_info["saved_checkpoints"].append({
                "epoch": epoch+1,
                "path": str(checkpoint_path),
                "val_acc": val_acc,
                "train_loss": avg_train_loss
            })
        
        # Save/update the best model if this epoch outperforms previous ones
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = model_manager.save_model(model, metrics, epoch=epoch+1, mode="best")
            experiment_info["best_model"] = {
                "epoch": epoch+1,
                "path": str(best_model_path),
                "val_acc": val_acc,
                "train_loss": avg_train_loss
            }
            print(f"🏆 New best model updated: {val_acc:.2f}%")
    
    # Save experiment summary info to preserve all details of this experiment.
    model_manager.save_experiment_info(experiment_info)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

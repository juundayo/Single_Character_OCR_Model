# ----------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------------------------------------------------------#

class ProgressTracker:
    def __init__(self, json_path='training_progress.json'):
        self.json_path = json_path
        self.progress_data = self.load_progress()
    
    def load_progress(self):
        """Load existing progress or create new structure"""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            except:
                print("Warning: Could not load progress file, creating new one")
        
        return {
            "training_start_time": datetime.now().isoformat(),
            "current_epoch": 0,
            "total_epochs": 0,
            "best_validation_accuracy": 0.0,
            "best_epoch": 0,
            "epochs": [],
            "final_test_accuracy": 0.0,
            "training_completed": False,
            "model_parameters": {},
            "training_time_seconds": 0
        }
    
    def update_epoch(self, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, 
                    epoch_time, learning_rate, model_parameters=None):
        """Update progress after each epoch"""
        epoch_data = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "epoch_time_seconds": float(epoch_time),
            "learning_rate": float(learning_rate),
            "timestamp": datetime.now().isoformat()
        }
        
        # Main progress data.
        self.progress_data["current_epoch"] = epoch
        self.progress_data["total_epochs"] = total_epochs
        
        # Adding epoch data.
        if len(self.progress_data["epochs"]) < epoch:
            self.progress_data["epochs"].append(epoch_data)
        else:
            self.progress_data["epochs"][epoch-1] = epoch_data
        
        # Updating the best accuracy.
        if val_acc > self.progress_data["best_validation_accuracy"]:
            self.progress_data["best_validation_accuracy"] = float(val_acc)
            self.progress_data["best_epoch"] = epoch
        
        # Updating model parameters (if provided).
        if model_parameters:
            self.progress_data["model_parameters"] = model_parameters
        
        self.save_progress()
    
    def update_training_complete(self, test_accuracy, total_training_time):
        """Mark training as completed"""
        self.progress_data["training_completed"] = True
        self.progress_data["final_test_accuracy"] = float(test_accuracy)
        self.progress_data["training_time_seconds"] = float(total_training_time)
        self.progress_data["training_end_time"] = datetime.now().isoformat()
        self.save_progress()
    
    def save_progress(self):
        """Save progress to JSON file"""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress to {self.json_path}: {e}")
    
    def print_progress_summary(self):
        """Print a summary of current progress"""
        if self.progress_data["epochs"]:
            latest = self.progress_data["epochs"][-1]
            print(f"\n=== TRAINING PROGRESS ===")
            print(f"Epoch: {self.progress_data['current_epoch']}/{self.progress_data['total_epochs']}")
            print(f"Latest - Train Loss: {latest['train_loss']:.4f}, Train Acc: {latest['train_accuracy']:.2f}%")
            print(f"Latest - Val Loss: {latest['validation_loss']:.4f}, Val Acc: {latest['validation_accuracy']:.2f}%")
            print(f"Best - Val Acc: {self.progress_data['best_validation_accuracy']:.2f}% (epoch {self.progress_data['best_epoch']})")
            print("=" * 30)

# ----------------------------------------------------------------------------#

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

# ----------------------------------------------------------------------------#

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, D)
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + h
        h = self.mlp(self.norm2(x))
        x = x + h
        return x
    
# ----------------------------------------------------------------------------#

class HybridOCR(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=2): 
        super(HybridOCR, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),  
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)  
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  #
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  
        )

        self.proj = nn.Conv2d(64, embed_dim, kernel_size=1)

        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim))  # 7x7=49

        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # CNN features.
        x = self.stem(x)
        x = self.conv_layers(x)
        
        # Project to embeddings.
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Adding positional encoding.
        x = x + self.pos_embedding
        
        # Transformer.
        x = self.transformer(x)
        
        # Global average pooling.
        x = x.mean(dim=1)
        
        # Classification.
        return self.classifier(x)

# ----------------------------------------------------------------------------#

class DynamicAugmentations:
    def __init__(self, device='cuda'):
        self.device = device
        self.augmentations = [
            ("RandomAffine", transforms.RandomAffine(
                degrees=2, translate=(0.02, 0.02), shear = 2,
                interpolation=transforms.InterpolationMode.BILINEAR
            )),
            ("RandomPerspective", transforms.RandomPerspective(
                distortion_scale=0.3, p=0.4,
                interpolation=transforms.InterpolationMode.BILINEAR
            )),
            ("GaussianBlur", transforms.GaussianBlur(kernel_size=3, 
                                                     sigma=(0.1, 1.0))),
            ("ColorJitter+Noise", transforms.Compose([
                transforms.ColorJitter(contrast=(0.5, 1.9)),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03)
            ]))
        ]
    
    def __call__(self, batch):
        """
        Applying dynamic augmentations to a batch
        with 50% probability for each augmentation.
        """
        augmented_batch = batch.clone()
        
        for aug_name, augmentation in self.augmentations:
            if random.random() < 0.5:  # 50% chance to apply each augmentation
                augmented_batch = augmentation(augmented_batch)
        
        return augmented_batch

# ----------------------------------------------------------------------------#

def load_char_mapping(mapping_path):
    """
    Loads character to ID mapping from the 
    JSON file and normalizes to contiguous IDs.
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        raw_mapping = json.load(f)

    # Build a sorted list of unique class IDs
    unique_ids = sorted(set(raw_mapping.values()))
    id_remap = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

    # Rebuild mapping: char -> contiguous ID
    char_to_id = {char: id_remap[old_id] for char, old_id in raw_mapping.items()}

    # Build reverse mapping for folder lookup (original numeric ID -> character)
    id_to_char = {old_id: char for char, old_id in raw_mapping.items()}

    print(f"Loaded character mapping with {len(char_to_id)} classes (remapped contiguously).")
    print("Example mapping:", list(char_to_id.items())[:10])  
    return char_to_id, id_to_char

# ----------------------------------------------------------------------------#

class OCRDataset(Dataset):
    def __init__(self, root_dir, char_to_id, id_to_char, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.samples = []

        print(f"Loading dataset from: {root_dir}")

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                try:
                    class_id = int(class_dir)  
                    if class_id in self.id_to_char:
                        char_label = self.id_to_char[class_id]
                        mapped_class_id = self.char_to_id[char_label]
                    else:
                        print(f"Warning: Class ID {class_id} not found in mapping")
                        continue
                except ValueError:
                    print(f"Skipping folder {class_dir}, not numeric")
                    continue

                image_count = 0
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        img_path = os.path.join(class_path, img_file)
                        self.samples.append((img_path, mapped_class_id))
                        image_count += 1

                print(f"  - Class {class_dir} -> {char_label} (mapped to {mapped_class_id}): {image_count} images")

        print(f"Total samples loaded: {len(self.samples)}")
        if len(self.samples) == 0:
            print("WARNING: No samples found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return image, label
    
# ----------------------------------------------------------------------------#

class OCRTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', progress_tracker=None, 
                 start_epoch=0, initial_best_acc=0.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.augmenter = DynamicAugmentations(device)
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.start_epoch = start_epoch
        self.best_acc = initial_best_acc
        
        # Loss and optimizer.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history.
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.start_time = time.time()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model and optimizer state from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restoring the training state.
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']
        if 'val_acc' in checkpoint:
            self.best_acc = checkpoint['val_acc']
        
        print(f"Loaded checkpoint from epoch {self.start_epoch} with validation accuracy: {self.best_acc:.2f}%")
        
        # Restoring the scheduler state if available.
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Applying dynamic augmentations with 50% probability for the entire batch.
            if random.random() < 0.5:
                data = self.augmenter(data)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Updating the progress bar.
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} Validation')
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, save_path='best_model.pth', resume_from_checkpoint=None):
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        print(f"Starting training from epoch {self.start_epoch + 1} to {self.start_epoch + num_epochs}...")
        
        # Initializing the model parameters for progress tracking.
        model_parameters = {
            "num_classes": self.model.classifier[-1].out_features,
            "embed_dim": self.model.proj.out_channels,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        for epoch in range(self.start_epoch + 1, self.start_epoch + num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase.
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase.
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Updating the learning rate.
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch}/{self.start_epoch + num_epochs} - Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
            
            # Updating the progress tracker.
            self.progress_tracker.update_epoch(
                epoch=epoch,
                total_epochs=self.start_epoch + num_epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time=epoch_time,
                learning_rate=current_lr,
                model_parameters=model_parameters
            )
            
            # Printing the progress summary.
            self.progress_tracker.print_progress_summary()
            
            # Saving the model if best.
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_path)
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        total_training_time = time.time() - self.start_time
        print(f'Training completed. Best validation accuracy: {self.best_acc:.2f}%')
        print(f'Total training time: {total_training_time:.2f} seconds')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'total_training_time': total_training_time
        }

# ----------------------------------------------------------------------------#

def create_data_loaders(base_path, char_to_id, id_to_char, batch_size=16, img_size=(64, 128)):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = OCRDataset(os.path.join(base_path, 'Training'), char_to_id, id_to_char, transform=train_transform)
    val_dataset   = OCRDataset(os.path.join(base_path, 'Validation'), char_to_id, id_to_char, transform=val_test_transform)
    test_dataset  = OCRDataset(os.path.join(base_path, 'Testing'), char_to_id, id_to_char, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

# ----------------------------------------------------------------------------#

def load_existing_progress(progress_file='ocr_training_progress.json'):
    """Loads the existing training progress to determine the starting point."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        if progress_data.get('epochs'):
            last_epoch = progress_data['epochs'][-1]
            print(f"Found existing training progress up to epoch {last_epoch['epoch']}")
            print(f"Last validation accuracy: {last_epoch['validation_accuracy']:.2f}%")
            return last_epoch['epoch'], progress_data['best_validation_accuracy']
    
    return 0, 0.0

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths!
    base_dataset_path = r"C:\Users\bgat\OCR\SplitDataset"
    char_mapping_path = r"C:\Users\bgat\OCR\Full_Dataset_1\char_to_id.json"
    
    # Model parameters.
    batch_size = 32
    additional_epochs = 25
    
    # Resume training?
    resume_training = input("Do you want to resume training from a saved model? (y/n): ").lower().strip() == 'y'
    
    # Initializing the progress tracker.
    progress_tracker = ProgressTracker('ocr_training_progress.json')
    
    char_to_id, id_to_char = load_char_mapping(char_mapping_path)
    num_classes = len(char_to_id)
    print(f"Number of classes: {num_classes}")

    train_loader, val_loader, test_loader = create_data_loaders(
        base_dataset_path, char_to_id, id_to_char, batch_size=batch_size
    )
        
    # Model creation.
    model = HybridOCR(num_classes=num_classes)
    print(f"Model created with {num_classes} output classes")
    
    # Determining the starting point.
    if resume_training:
        checkpoint_path = 'OCR_model.pth'
        if os.path.exists(checkpoint_path):
            # Checkpoint loading.
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resuming training from epoch {start_epoch + 1} with best accuracy: {best_acc:.2f}%")
            
            # Create trainer with resume parameters
            trainer = OCRTrainer(model, train_loader, val_loader, device, progress_tracker, 
                               start_epoch=start_epoch, initial_best_acc=best_acc)
            
            # Train for additional epochs
            history = trainer.train(num_epochs=additional_epochs, save_path='OCR_model.pth', 
                                  resume_from_checkpoint=checkpoint_path)
        else:
            print(f"Checkpoint file {checkpoint_path} not found. Starting fresh training.")
            trainer = OCRTrainer(model, train_loader, val_loader, device, progress_tracker)
            history = trainer.train(num_epochs=additional_epochs, save_path='OCR_model.pth')
    else:
        # Training.
        start_epoch, best_acc = load_existing_progress()
        trainer = OCRTrainer(model, train_loader, val_loader, device, progress_tracker, 
                           start_epoch=start_epoch, initial_best_acc=best_acc)
        history = trainer.train(num_epochs=additional_epochs, save_path='OCR_model.pth')
    
    # Testing the model!
    print("\nTesting the model...")
    checkpoint = torch.load('OCR_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_accuracy = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Updating the progress tracker with the final results.
    progress_tracker.update_training_complete(test_accuracy, history['total_training_time'])
    
    # Plotting the training history.
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.plot(history['val_accs'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print(f"\nTraining progress saved to: {progress_tracker.json_path}")

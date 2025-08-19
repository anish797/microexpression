import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import timm
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class CASMEMultiFrameDataset(Dataset):
    """Dataset that loads multiple frames per video (sequences)"""
    
    def __init__(self, excel_path, base_path, num_frames=8, img_size=224, transform=None):
        """
        Args:
            excel_path: Path to CASME excel file
            base_path: Path to /casme/cropped/ directory
            num_frames: Number of frames to extract per video sequence
            img_size: Image size (224 for EfficientNet)
        """
        print("🔄 Loading multi-frame dataset...")
        
        # Read Excel
        self.df = pd.read_excel(excel_path, header=None)
        self.df.columns = ['participant', 'video_name', 'start_frame', 'apex_frame', 
                          'end_frame', 'AUs', 'estimated_emotion', 'expression_type', 'self_reported_emotion']
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform if transform else self.get_default_transform()
        
        # Extract emotion instances
        self.df['emotion_instance'] = self.df['video_name'].str.replace(r'_\d+$', '', regex=True)
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['emotion_instance'])
        
        # Use all data
        self.data_df = self.df.copy().reset_index(drop=True)
        self.data_labels = self.label_encoder.transform(self.data_df['emotion_instance'])
        
        print(f"✅ Loaded {len(self.data_df)} samples")
        print(f"✅ Found {len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}")
        print(f"✅ Using {num_frames} frames per video")
        
    def get_default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_augmented_transform(self):
        """Data augmentation transforms for training"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            # Data augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def fix_zero_end_frames(self):
        """Fix end_frame = 0 by using apex_frame"""
        zero_mask = self.data_df['end_frame'] == 0
        fixed_count = zero_mask.sum()
        
        self.data_df.loc[zero_mask, 'end_frame'] = self.data_df.loc[zero_mask, 'apex_frame']
        
        print(f"✅ Fixed {fixed_count} videos with end_frame = 0")
        return self
    
    def load_frame_sequence(self, participant, video_name, start_frame, apex_frame, end_frame):
        """Load sequence of frames from start to end"""
        participant_path = os.path.join(self.base_path, str(participant))
        video_path = os.path.join(participant_path, video_name)
        
        # Get all available frames in the range
        available_frames = []
        for frame_num in range(start_frame, end_frame + 1):
            # Try different naming patterns
            for pattern in [f"img{frame_num}.jpg", f"img{frame_num:05d}.jpg", f"img{frame_num:04d}.jpg"]:
                frame_path = os.path.join(video_path, pattern)
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        available_frames.append(frame)
                        break
        
        # If no frames found, return zeros
        if not available_frames:
            print(f"⚠️  No frames found in {video_path} for range {start_frame}-{end_frame}")
            return torch.zeros((self.num_frames, 3, self.img_size, self.img_size))
        
        # Sample frames to get exactly num_frames
        sampled_frames = self.sample_frames(available_frames, self.num_frames)
        
        # Transform each frame
        transformed_frames = []
        for frame in sampled_frames:
            transformed_frames.append(self.transform(frame))
        
        return torch.stack(transformed_frames)  # Shape: (num_frames, 3, 224, 224)
    
    def sample_frames(self, frames, target_count):
        """Sample exactly target_count frames from available frames"""
        if len(frames) == target_count:
            return frames
        elif len(frames) > target_count:
            # Evenly sample frames
            indices = np.linspace(0, len(frames)-1, target_count, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Duplicate frames to reach target count
            sampled = frames.copy()
            while len(sampled) < target_count:
                sampled.extend(frames)  # Repeat the sequence
            return sampled[:target_count]  # Trim to exact count
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Load frame sequence
        frame_sequence = self.load_frame_sequence(
            row['participant'], 
            row['video_name'], 
            row['start_frame'],
            row['apex_frame'],
            row['end_frame']
        )
        
        label = self.data_labels[idx]
        return frame_sequence, torch.tensor(label, dtype=torch.long)

class MultiFrameEfficientNet(nn.Module):
    """EfficientNet that processes multiple frames with temporal averaging"""
    
    def __init__(self, num_classes, num_frames=8):
        super(MultiFrameEfficientNet, self).__init__()
        
        self.num_frames = num_frames
        
        # Load pre-trained EfficientNet-B0 (remove classification head)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.feature_dim = self.efficientnet.num_features  # 1280 for EfficientNet-B0
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"✅ Created MultiFrame EfficientNet")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Frames per video: {num_frames}")
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, 3, 224, 224)
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        # Reshape to process all frames together
        x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
        
        # Extract features using EfficientNet
        features = self.efficientnet(x)  # Shape: (batch_size * num_frames, feature_dim)
        
        # Reshape back to sequences
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, feature_dim)
        
        # Temporal aggregation - simple averaging
        temporal_features = features.mean(dim=1)  # (batch_size, feature_dim)
        
        # Classification
        output = self.classifier(temporal_features)
        return output

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3, device='cuda', class_weights=None):
    """Training function with class balancing"""
    model = model.to(device)
    
    # Use weighted loss for class balancing
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✅ Using weighted loss for class balancing")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
    
    train_losses, val_accuracies = [], []
    best_val_acc = 0.0
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        
        for batch_idx, (videos, labels) in enumerate(train_pbar):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_multiframe_efficientnet.pth')
            print(f'🎯 New best model saved! Validation Accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    print(f'\n🏆 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    return train_losses, val_accuracies

def test_multiframe_dataset(dataset, num_samples=3):
    """Test multi-frame dataset loading"""
    print(f"\n🧪 Testing multi-frame dataset with {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            frames, label = dataset[i]
            class_name = dataset.label_encoder.classes_[label]
            print(f"  Sample {i}: Frames shape: {frames.shape}, Label: {label} ({class_name})")
            
            # Check if frames contain actual data (not all zeros)
            non_zero_pixels = (frames != 0).sum().item()
            total_pixels = frames.numel()
            print(f"    Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Error loading sample {i}: {e}")
            return False
    
    print("✅ Multi-frame dataset test passed!")
    return True

def calculate_class_weights(labels, device):
    """Calculate class weights for balancing"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get class distribution
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    print(f"\n📊 Class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"   Class {cls}: {count} samples")
    
    # Calculate balanced weights
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"\n⚖️  Class weights:")
    for cls, weight in zip(unique_classes, class_weights):
        print(f"   Class {cls}: {weight:.3f}")
    
    return class_weights

def create_augmented_dataset(dataset, train_indices):
    """Create dataset with augmentation for training"""
    
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, use_augmentation=True):
            self.base_dataset = base_dataset
            self.indices = indices
            self.use_augmentation = use_augmentation
            
            if use_augmentation:
                self.aug_transform = base_dataset.get_augmented_transform()
            
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            row = self.base_dataset.data_df.iloc[real_idx]
            
            # Load frame sequence
            participant_path = os.path.join(self.base_dataset.base_path, str(row['participant']))
            video_path = os.path.join(participant_path, row['video_name'])
            
            # Get available frames
            available_frames = []
            for frame_num in range(row['start_frame'], row['end_frame'] + 1):
                for pattern in [f"img{frame_num}.jpg", f"img{frame_num:05d}.jpg", f"img{frame_num:04d}.jpg"]:
                    frame_path = os.path.join(video_path, pattern)
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            available_frames.append(frame)
                            break
            
            if not available_frames:
                # Return zeros if no frames found
                frame_sequence = torch.zeros((self.base_dataset.num_frames, 3, self.base_dataset.img_size, self.base_dataset.img_size))
            else:
                # Sample frames
                sampled_frames = self.base_dataset.sample_frames(available_frames, self.base_dataset.num_frames)
                
                # Apply transforms (with or without augmentation)
                transformed_frames = []
                for frame in sampled_frames:
                    if self.use_augmentation:
                        transformed_frames.append(self.aug_transform(frame))
                    else:
                        transformed_frames.append(self.base_dataset.transform(frame))
                
                frame_sequence = torch.stack(transformed_frames)
            
            label = self.base_dataset.data_labels[real_idx]
            return frame_sequence, torch.tensor(label, dtype=torch.long)
    
    return AugmentedDataset(dataset, train_indices, use_augmentation=True)

def fix_zero_end_frames(dataset):
    """Fix end_frame = 0 by using apex_frame"""
    zero_mask = dataset.data_df['end_frame'] == 0
    fixed_count = zero_mask.sum()
    
    dataset.data_df.loc[zero_mask, 'end_frame'] = dataset.data_df.loc[zero_mask, 'apex_frame']
    
    print(f"✅ Fixed {fixed_count} videos with end_frame = 0")
    return dataset
    """Fix end_frame = 0 by using apex_frame"""
    zero_mask = dataset.data_df['end_frame'] == 0
    fixed_count = zero_mask.sum()
    
    dataset.data_df.loc[zero_mask, 'end_frame'] = dataset.data_df.loc[zero_mask, 'apex_frame']
    
    print(f"✅ Fixed {fixed_count} videos with end_frame = 0")
    return dataset

if __name__ == "__main__":
    # Configuration
    EXCEL_PATH = "../casme/annotations.xlsx"
    BASE_PATH = "../casme/cropped/"
    BATCH_SIZE = 8
    NUM_EPOCHS = 35  # 🧪 TESTING: Single epoch for quick testing
    LEARNING_RATE = 1e-3
    NUM_FRAMES = 8
    USE_DATA_AUGMENTATION = True
    USE_CLASS_BALANCING = True
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("🚀 STEP 2: MULTI-FRAME EFFICIENTNET (TESTING)")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Frames per video: {NUM_FRAMES}")
    print(f"Data augmentation: {USE_DATA_AUGMENTATION}")
    print(f"Class balancing: {USE_CLASS_BALANCING}")
    print(f"🧪 TESTING MODE: {NUM_EPOCHS} epoch only")
    
    # Step 2.1: Create multi-frame dataset
    print("\n📁 Step 2.1: Creating multi-frame dataset...")
    dataset = CASMEMultiFrameDataset(EXCEL_PATH, BASE_PATH, num_frames=NUM_FRAMES)
    dataset = fix_zero_end_frames(dataset)
    
    # Test multi-frame dataset
    if not test_multiframe_dataset(dataset):
        print("❌ Multi-frame dataset test failed! Fix issues before continuing.")
        exit(1)
    
    # Step 2.2: Create multi-frame model
    print("\n🤖 Step 2.2: Creating multi-frame model...")
    num_classes = len(dataset.label_encoder.classes_)
    model = MultiFrameEfficientNet(num_classes, num_frames=NUM_FRAMES)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model has {total_params:,} parameters")
    
    # Step 2.3: Create data splits
    print("\n📦 Step 2.3: Creating data splits...")
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=dataset.data_labels
    )
    
    print(f"✅ Train samples: {len(train_indices)}")
    print(f"✅ Validation samples: {len(val_indices)}")
    
    # Step 2.4: Calculate class weights for balancing
    class_weights = None
    if USE_CLASS_BALANCING:
        print("\n⚖️  Step 2.4: Calculating class weights...")
        train_labels = [dataset.data_labels[i] for i in train_indices]
        class_weights = calculate_class_weights(train_labels, DEVICE)
    
    # Step 2.5: Create data loaders with augmentation
    print(f"\n📦 Step 2.5: Creating data loaders...")
    
    if USE_DATA_AUGMENTATION:
        print("✅ Using data augmentation for training")
        train_dataset = create_augmented_dataset(dataset, train_indices)
    else:
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Step 2.6: Test model forward pass
    print("\n🧪 Step 2.6: Testing model forward pass...")
    model = model.to(DEVICE)
    
    try:
        sample_videos, sample_labels = next(iter(train_loader))
        sample_videos = sample_videos.to(DEVICE)
        
        print(f"  Input shape: {sample_videos.shape}")
        
        with torch.no_grad():
            outputs = model(sample_videos)
            predictions = torch.max(outputs, 1)[1]
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Predictions: {predictions}")
        print("✅ Model forward pass test passed!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        exit(1)
    
    # Step 2.7: Start training
    print("\n" + "="*60)
    print("🏋️ STARTING MULTI-FRAME TRAINING")
    print("="*60)
    
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE, 
        device=DEVICE,
        class_weights=class_weights
    )
    
    # Results
    best_acc = max(val_accuracies) if val_accuracies else 0
    print(f"\n📊 STEP 2 RESULTS (1 EPOCH TEST):")
    print(f"   Validation accuracy: {best_acc:.2f}%")
    print(f"   Training loss: {train_losses[-1]:.4f}")
    
    # Analysis
    print(f"\n📈 QUICK ANALYSIS:")
    if best_acc > 25:
        print(f"   ✅ Good start! {best_acc:.1f}% > 25% (better than random)")
        print(f"   🚀 Ready for full training (set NUM_EPOCHS = 20)")
        if best_acc > 40:
            print(f"   🎉 Excellent! Already beating single-frame baseline")
    else:
        print(f"   ⚠️  Low accuracy ({best_acc:.1f}%), check:")
        print(f"   - Data loading issues")
        print(f"   - Model architecture")
        print(f"   - Learning rate too high/low")
    
    print(f"\n🔧 TO RUN FULL TRAINING:")
    print(f"   Set NUM_EPOCHS = 20 and run again")
    print(f"   Expected full training time: ~40-60 minutes")
    
    # Plot results (even for 1 epoch)
    if train_losses:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'o-')
        plt.title('Training Loss (1 Epoch Test)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'o-')
        plt.title('Validation Accuracy (1 Epoch Test)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('step2_test_results.png')
        print("📈 Test results saved as 'step2_test_results.png'")
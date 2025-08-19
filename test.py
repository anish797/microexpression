import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import timm
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class CASMEMultiFrameDataset(Dataset):
    """Dataset for validation - same as training"""
    
    def __init__(self, excel_path, base_path, num_frames=8, img_size=224, transform=None):
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
        
        # Fix end_frame = 0
        self.fix_zero_end_frames()
        
    def fix_zero_end_frames(self):
        zero_mask = self.data_df['end_frame'] == 0
        self.data_df.loc[zero_mask, 'end_frame'] = self.data_df.loc[zero_mask, 'apex_frame']
        
    def get_default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def sample_frames(self, frames, target_count):
        if len(frames) == target_count:
            return frames
        elif len(frames) > target_count:
            indices = np.linspace(0, len(frames)-1, target_count, dtype=int)
            return [frames[i] for i in indices]
        else:
            sampled = frames.copy()
            while len(sampled) < target_count:
                sampled.extend(frames)
            return sampled[:target_count]
    
    def load_frame_sequence(self, participant, video_name, start_frame, apex_frame, end_frame):
        participant_path = os.path.join(self.base_path, str(participant))
        video_path = os.path.join(participant_path, video_name)
        
        available_frames = []
        for frame_num in range(start_frame, end_frame + 1):
            for pattern in [f"img{frame_num}.jpg", f"img{frame_num:05d}.jpg", f"img{frame_num:04d}.jpg"]:
                frame_path = os.path.join(video_path, pattern)
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        available_frames.append(frame)
                        break
        
        if not available_frames:
            return torch.zeros((self.num_frames, 3, self.img_size, self.img_size))
        
        sampled_frames = self.sample_frames(available_frames, self.num_frames)
        
        transformed_frames = []
        for frame in sampled_frames:
            transformed_frames.append(self.transform(frame))
        
        return torch.stack(transformed_frames)
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
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
    """Same model as training"""
    
    def __init__(self, num_classes, num_frames=8):
        super(MultiFrameEfficientNet, self).__init__()
        
        self.num_frames = num_frames
        
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.feature_dim = self.efficientnet.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
        features = self.efficientnet(x)
        features = features.view(batch_size, num_frames, -1)
        
        # Temporal aggregation - simple averaging
        temporal_features = features.mean(dim=1)
        
        output = self.classifier(temporal_features)
        return output

def load_model(model_path, num_classes, device, num_frames=8):
    """Load trained model"""
    print(f"📥 Loading model from: {model_path}")
    
    model = MultiFrameEfficientNet(num_classes, num_frames=num_frames)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def comprehensive_validation(model, test_loader, dataset, device, save_plots=True):
    """Comprehensive validation with detailed metrics"""
    print(f"\n🧪 Starting comprehensive validation...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # Get predictions
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Validating"):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Class names
    class_names = dataset.label_encoder.classes_
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Print per-class metrics
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name:>10}: Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}, "
                  f"Support: {int(metrics['support'])}")
    
    # Macro and weighted averages
    print(f"\n   {'Macro Avg':>10}: Precision: {report['macro avg']['precision']:.3f}, "
          f"Recall: {report['macro avg']['recall']:.3f}, F1: {report['macro avg']['f1-score']:.3f}")
    print(f"   {'Weighted Avg':>10}: Precision: {report['weighted avg']['precision']:.3f}, "
          f"Recall: {report['weighted avg']['recall']:.3f}, F1: {report['weighted avg']['f1-score']:.3f}")
    
    if save_plots:
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Per-class accuracy
        plt.subplot(2, 2, 2)
        class_accuracies = [report[class_name]['recall'] for class_name in class_names if class_name in report]
        bars = plt.bar(class_names, class_accuracies)
        plt.title('Per-Class Recall (Accuracy)')
        plt.ylabel('Recall')
        plt.xticks(rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if class_accuracies[i] > 0.8:
                bar.set_color('green')
            elif class_accuracies[i] > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Class distribution
        plt.subplot(2, 2, 3)
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        class_counts = [counts[i] for i in unique_labels]
        class_labels_for_plot = [class_names[i] for i in unique_labels]
        
        plt.pie(class_counts, labels=class_labels_for_plot, autopct='%1.1f%%')
        plt.title('Test Set Class Distribution')
        
        # Prediction confidence distribution
        plt.subplot(2, 2, 4)
        max_probs = np.max(all_probabilities, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Max Probability')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('validation_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Validation plots saved as 'validation_analysis.png'")
        plt.show()
    
    # Find best and worst performing classes
    class_f1_scores = [(class_name, report[class_name]['f1-score']) 
                       for class_name in class_names if class_name in report]
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 BEST PERFORMING CLASSES:")
    for i, (class_name, f1) in enumerate(class_f1_scores[:3]):
        print(f"   {i+1}. {class_name}: F1 = {f1:.3f}")
    
    print(f"\n😞 WORST PERFORMING CLASSES:")
    for i, (class_name, f1) in enumerate(class_f1_scores[-3:]):
        print(f"   {len(class_f1_scores)-2+i}. {class_name}: F1 = {f1:.3f}")
    
    # Error analysis
    print(f"\n🔍 ERROR ANALYSIS:")
    errors = all_labels != all_predictions
    error_count = np.sum(errors)
    print(f"   Total errors: {error_count}/{len(all_labels)} ({error_count/len(all_labels)*100:.1f}%)")
    
    if error_count > 0:
        # Most common misclassifications
        error_pairs = []
        for true_label, pred_label in zip(all_labels[errors], all_predictions[errors]):
            error_pairs.append((class_names[true_label], class_names[pred_label]))
        
        from collections import Counter
        common_errors = Counter(error_pairs).most_common(5)
        
        print(f"   Most common misclassifications:")
        for (true_class, pred_class), count in common_errors:
            print(f"     {true_class} → {pred_class}: {count} times")
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'classification_report': report,
        'confusion_matrix': cm
    }

def validate_trained_model():
    """Main validation function with multiple split options"""
    # Configuration
    EXCEL_PATH = "../casme/annotations.xlsx"
    BASE_PATH = "../casme/cropped/"
    MODEL_PATH = "best_multiframe_efficientnet.pth"  # Update this path
    BATCH_SIZE = 8
    NUM_FRAMES = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("🧪 MODEL VALIDATION SCRIPT")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    
    # Load dataset
    print(f"\n📁 Loading dataset...")
    dataset = CASMEMultiFrameDataset(EXCEL_PATH, BASE_PATH, num_frames=NUM_FRAMES)
    
    print(f"✅ Loaded {len(dataset)} samples")
    print(f"✅ Found {len(dataset.label_encoder.classes_)} classes: {dataset.label_encoder.classes_}")
    
    print(f"\n🎯 Choose validation strategy:")
    print(f"1. Different random split (recommended for true generalization test)")
    print(f"2. Leave-One-Subject-Out (LOSO) validation")
    print(f"3. Holdout specific participants")
    print(f"4. Original training split (same as training)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Different random split with different random state
        print(f"\n🎲 Using DIFFERENT random split (random_state=999)")
        train_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.2, 
            random_state=55,  # DIFFERENT random state
            stratify=dataset.data_labels
        )
        test_name = "Different Random Split"
        
    elif choice == "2":
        # Leave-One-Subject-Out (LOSO)
        participants = dataset.data_df['participant'].unique()
        print(f"\n👥 Available participants: {participants}")
        test_participant = input(f"Enter participant ID to hold out for testing: ").strip()
        
        try:
            test_participant = int(test_participant)
            if test_participant not in participants:
                print(f"❌ Participant {test_participant} not found!")
                return
            
            test_indices = dataset.data_df[dataset.data_df['participant'] == test_participant].index.tolist()
            train_indices = dataset.data_df[dataset.data_df['participant'] != test_participant].index.tolist()
            test_name = f"LOSO (Participant {test_participant})"
            
        except ValueError:
            print("❌ Invalid participant ID!")
            return
            
    elif choice == "3":
        # Holdout specific participants
        participants = dataset.data_df['participant'].unique()
        print(f"\n👥 Available participants: {participants}")
        test_participants = input(f"Enter participant IDs to hold out (comma-separated): ").strip()
        
        try:
            test_participants = [int(p.strip()) for p in test_participants.split(',')]
            missing = [p for p in test_participants if p not in participants]
            if missing:
                print(f"❌ Participants not found: {missing}")
                return
                
            test_mask = dataset.data_df['participant'].isin(test_participants)
            test_indices = dataset.data_df[test_mask].index.tolist()
            train_indices = dataset.data_df[~test_mask].index.tolist()
            test_name = f"Holdout Participants {test_participants}"
            
        except ValueError:
            print("❌ Invalid participant IDs!")
            return
            
    elif choice == "4":
        # Original training split
        print(f"\n📚 Using SAME random split as training (random_state=42)")
        train_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.2, 
            random_state=42,  # SAME random state as training
            stratify=dataset.data_labels
        )
        test_name = "Original Training Split"
        
    else:
        print("❌ Invalid choice!")
        return
    
    # Create test dataset
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n✅ Test strategy: {test_name}")
    print(f"✅ Test samples: {len(test_indices)}")
    print(f"✅ Train samples: {len(train_indices)}")
    
    # Show test set class distribution
    test_labels = [dataset.data_labels[i] for i in test_indices]
    unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
    print(f"\n📊 Test set class distribution:")
    for label_idx, count in zip(unique_test_labels, test_counts):
        class_name = dataset.label_encoder.classes_[label_idx]
        print(f"   {class_name}: {count} samples")
    
    # Load model
    num_classes = len(dataset.label_encoder.classes_)
    model = load_model(MODEL_PATH, num_classes, DEVICE, NUM_FRAMES)
    
    if model is None:
        print("❌ Failed to load model. Please check the model path.")
        return
    
    # Run comprehensive validation
    results = comprehensive_validation(model, test_loader, dataset, DEVICE)
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"🎯 FINAL VALIDATION SUMMARY")
    print(f"="*60)
    print(f"📊 Test Strategy: {test_name}")
    print(f"📊 Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"📁 Test samples: {len(test_indices)}")
    print(f"🎯 Classes: {len(dataset.label_encoder.classes_)}")
    print(f"💾 Model: {MODEL_PATH}")
    
    # Performance rating
    if results['accuracy'] > 0.8:
        print(f"🏆 EXCELLENT performance! (>80%)")
    elif results['accuracy'] > 0.7:
        print(f"✅ GOOD performance! (70-80%)")
    elif results['accuracy'] > 0.6:
        print(f"⚠️  FAIR performance (60-70%)")
    else:
        print(f"❌ POOR performance (<60%)")
    
    return results

if __name__ == "__main__":
    results = validate_trained_model()
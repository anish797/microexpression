import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import argparse
from PIL import Image
import timm
from torchvision import transforms
import matplotlib.pyplot as plt

class MultiFrameEfficientNet(nn.Module):
    """Same model architecture as training"""
    
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

class EmotionInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.img_size = 224
        self.num_frames = 8
        
        # CASME emotion classes (update based on your training)
        self.emotion_classes = [
            'anger1', 'anger2', 'disgust1', 'disgust2', 
            'happy1', 'happy2', 'happy3', 'happy4', 'happy5'
        ]
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """Load the trained model"""
        print(f"📥 Loading model from: {model_path}")
        
        model = MultiFrameEfficientNet(
            num_classes=len(self.emotion_classes), 
            num_frames=self.num_frames
        )
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            print(f"✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path  # Already numpy array
            
            # Convert to PIL Image for transforms
            pil_image = Image.fromarray(image)
            
            # Apply transforms
            transformed = self.transform(pil_image)
            
            return transformed
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def create_frame_sequence(self, image_path):
        """Create a sequence of frames from a single image (duplicate frames)"""
        # For single image, duplicate it to create sequence
        frame = self.preprocess_image(image_path)
        if frame is None:
            return None
        
        # Create sequence by duplicating the frame
        frame_sequence = torch.stack([frame] * self.num_frames)
        
        return frame_sequence.unsqueeze(0)  # Add batch dimension
    
    def create_frame_sequence_from_multiple(self, image_paths):
        """Create frame sequence from multiple images"""
        frames = []
        
        for img_path in image_paths:
            frame = self.preprocess_image(img_path)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            return None
        
        # Pad or sample to get exactly num_frames
        if len(frames) < self.num_frames:
            # Duplicate frames to reach num_frames
            while len(frames) < self.num_frames:
                frames.extend(frames[:self.num_frames - len(frames)])
        elif len(frames) > self.num_frames:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        frame_sequence = torch.stack(frames)
        return frame_sequence.unsqueeze(0)  # Add batch dimension
    
    def predict_emotion(self, input_data):
        """Predict emotion from image(s)"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        # Handle different input types
        if isinstance(input_data, str):
            # Single image path
            if os.path.isfile(input_data):
                frame_sequence = self.create_frame_sequence(input_data)
            else:
                print(f"❌ File not found: {input_data}")
                return None
        elif isinstance(input_data, list):
            # Multiple image paths
            frame_sequence = self.create_frame_sequence_from_multiple(input_data)
        else:
            print("❌ Invalid input data type")
            return None
        
        if frame_sequence is None:
            return None
        
        # Move to device
        frame_sequence = frame_sequence.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(frame_sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        predicted_class = self.emotion_classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            (self.emotion_classes[idx.item()], prob.item()) 
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        return {
            'predicted_emotion': predicted_class,
            'confidence': confidence_score,
            'top3_predictions': top3_predictions,
            'all_probabilities': probabilities[0].cpu().numpy()
        }
    
    def visualize_prediction(self, image_path, prediction_result):
        """Visualize the prediction results"""
        if prediction_result is None:
            return
        
        # Load and display image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f"Input Image")
        plt.axis('off')
        
        # Display predictions
        plt.subplot(1, 2, 2)
        emotions = [pred[0] for pred in prediction_result['top3_predictions']]
        confidences = [pred[1] for pred in prediction_result['top3_predictions']]
        
        colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' for i in range(3)]
        bars = plt.bar(emotions, confidences, color=colors)
        
        plt.title('Top 3 Emotion Predictions')
        plt.ylabel('Confidence')
        plt.xticks(rotation=45)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"\n🎯 EMOTION PREDICTION RESULTS:")
        print(f"📷 Image: {os.path.basename(image_path)}")
        print(f"🏆 Predicted Emotion: {prediction_result['predicted_emotion']}")
        print(f"📊 Confidence: {prediction_result['confidence']:.4f} ({prediction_result['confidence']*100:.2f}%)")
        
        print(f"\n📋 Top 3 Predictions:")
        for i, (emotion, conf) in enumerate(prediction_result['top3_predictions']):
            print(f"   {i+1}. {emotion}: {conf:.4f} ({conf*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--images', nargs='+', help='Paths to multiple images (for sequence)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = EmotionInference(args.model)
    
    if args.image:
        # Single image inference
        print(f"🔍 Analyzing single image: {args.image}")
        result = inferencer.predict_emotion(args.image)
        
        if result:
            if args.visualize:
                inferencer.visualize_prediction(args.image, result)
            else:
                print(f"\n🎯 Predicted Emotion: {result['predicted_emotion']}")
                print(f"📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
    elif args.images:
        # Multiple images inference
        print(f"🔍 Analyzing image sequence: {len(args.images)} images")
        result = inferencer.predict_emotion(args.images)
        
        if result:
            print(f"\n🎯 Predicted Emotion: {result['predicted_emotion']}")
            print(f"📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print(f"\n📋 Top 3 Predictions:")
            for i, (emotion, conf) in enumerate(result['top3_predictions']):
                print(f"   {i+1}. {emotion}: {conf:.4f} ({conf*100:.2f}%)")
    
    else:
        print("❌ Please provide either --image or --images argument")

if __name__ == "__main__":
    # Example usage if run directly
    import sys
    
    if len(sys.argv) == 1:
        print("🚀 EMOTION RECOGNITION INFERENCE SCRIPT")
        print("="*50)
        print("Usage examples:")
        print("1. Single image:")
        print("   python inference.py --model best_multiframe_efficientnet.pth --image photo.jpg --visualize")
        print("\n2. Multiple images (sequence):")
        print("   python inference.py --model best_multiframe_efficientnet.pth --images img1.jpg img2.jpg img3.jpg")
        print("\n3. Quick test (if you have test image):")
        
        # Quick test if model and test image exist
        model_path = "best_multiframe_efficientnet.pth"
        if os.path.exists(model_path):
            print(f"\n✅ Found model: {model_path}")
            
            # You can add a quick test here with any available image
            test_image = input("Enter path to test image (or press Enter to skip): ").strip()
            if test_image and os.path.exists(test_image):
                inferencer = EmotionInference(model_path)
                result = inferencer.predict_emotion(test_image)
                if result:
                    inferencer.visualize_prediction(test_image, result)
        else:
            print(f"❌ Model not found: {model_path}")
    else:
        main()
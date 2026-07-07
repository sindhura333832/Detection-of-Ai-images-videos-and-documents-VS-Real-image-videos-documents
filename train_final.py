import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import pickle

print("="*70)
print("🔥 FINAL TRAINING - PURE PYTHON IMPLEMENTATION")
print("="*70)

class FinalTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'Mean', 'Std', 'Median', 'Min', 'Max',
            'EdgeDensity', 'EdgeStrength',
            'TextureMean', 'TextureStd',
            'NoiseMean', 'NoiseStd',
            'HighFreqRatio',
            'ColorVariance'
        ]
        
    def extract_features(self, image_path):
        """Extract simple but powerful features"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ===== STATISTICAL FEATURES =====
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            median_val = np.median(gray)
            min_val = np.min(gray)
            max_val = np.max(gray)
            
            # ===== EDGE FEATURES =====
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # ===== TEXTURE FEATURES =====
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_mean = np.mean(np.abs(laplacian))
            texture_std = np.std(laplacian)
            
            # ===== NOISE FEATURES =====
            blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            noise_mean = np.mean(np.abs(noise))
            noise_std = np.std(noise)
            
            # ===== FREQUENCY FEATURES =====
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            h, w = magnitude.shape
            center_h, center_w = h//2, w//2
            
            # Calculate energy in different frequency bands
            low_freq = np.mean(magnitude[center_h-30:center_h+30, center_w-30:center_w+30])
            total_freq = np.mean(magnitude)
            high_freq_ratio = low_freq / (total_freq + 1e-10)
            
            # ===== COLOR FEATURES =====
            if len(img.shape) == 3:
                b, g, r = cv2.split(img)
                color_variance = np.var(b) + np.var(g) + np.var(r)
                color_mean = (np.mean(b) + np.mean(g) + np.mean(r)) / 3
            else:
                color_variance = 0
                color_mean = mean_val
            
            # Combine all features
            features = [
                mean_val, std_val, median_val, min_val, max_val,
                edge_density, edge_strength,
                texture_mean, texture_std,
                noise_mean, noise_std,
                high_freq_ratio,
                color_variance
            ]
            
            return features
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def load_batch(self, folder_path, label, batch_size=1000):
        """Load images in batches to save memory"""
        print(f"\n📁 Loading from: {folder_path}")
        
        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   Found {len(files)} images")
        
        X_batch = []
        y_batch = []
        
        for i, filename in enumerate(files):
            image_path = os.path.join(folder_path, filename)
            features = self.extract_features(image_path)
            
            if features:
                X_batch.append(features)
                y_batch.append(label)
            
            if (i + 1) % batch_size == 0:
                print(f"   Loaded {i+1}/{len(files)} images")
                
        return np.array(X_batch), np.array(y_batch)
    
    def load_all_data(self):
        """Load all 120,000 images"""
        print("\n📂 LOADING ALL 120,000 IMAGES...")
        print("="*50)
        
        base_path = r"C:\Users\Ideapad\AI_Simple\cifake_dataset"
        
        # Define all folders
        folders = [
            (os.path.join(base_path, "train", "REAL"), 0, "Train REAL (40k)"),
            (os.path.join(base_path, "train", "FAKE"), 1, "Train AI (40k)"),
            (os.path.join(base_path, "test", "REAL"), 0, "Test REAL (20k)"),
            (os.path.join(base_path, "test", "FAKE"), 1, "Test AI (20k)")
        ]
        
        all_X = []
        all_y = []
        total = 0
        
        start_time = time.time()
        
        for folder_path, label, name in folders:
            if not os.path.exists(folder_path):
                print(f"❌ Not found: {folder_path}")
                continue
            
            X_batch, y_batch = self.load_batch(folder_path, label)
            
            if len(X_batch) > 0:
                all_X.extend(X_batch)
                all_y.extend(y_batch)
                total += len(X_batch)
                print(f"✅ {name}: {len(X_batch)} images")
            
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            print(f"   Progress: {total}/120,000 images ({rate:.1f} img/sec)\n")
        
        total_time = time.time() - start_time
        
        print("="*50)
        print(f"✅ TOTAL LOADED: {total} / 120,000 images")
        print(f"⏱️  Time: {total_time/60:.2f} minutes")
        print(f"⚡ Speed: {total/elapsed:.1f} images/second")
        
        return np.array(all_X), np.array(all_y)
    
    def train(self, X, y):
        """Train HistGradientBoosting (pure Python, no compilation)"""
        print("\n🧠 TRAINING HISTOGRAM GRADIENT BOOSTING...")
        print("="*50)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} images")
        print(f"Validation set: {len(X_val):,} images")
        
        # Create model (pure Python implementation)
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=15,
            learning_rate=0.1,
            random_state=42,
            verbose=1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        # Train
        start_time = time.time()
        print("\n🚀 Training started...")
        
        self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print("\n" + "="*50)
        print("📊 TRAINING RESULTS")
        print("="*50)
        print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%")
        print(f"⏱️  Training time: {train_time/60:.2f} minutes")
        
        # Detailed metrics
        print("\n📋 Classification Report:")
        print(classification_report(y_val, y_pred, 
                                   target_names=['REAL PHOTO', 'AI GENERATED']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"               Predicted")
        print(f"               REAL    AI")
        print(f"Actual REAL    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       AI      {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        return accuracy
    
    def save_model(self, filename):
        """Save model"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to: {filename}")
        file_size = os.path.getsize(filename) / (1024*1024)
        print(f"📦 File size: {file_size:.2f} MB")

# ===== MAIN =====
if __name__ == "__main__":
    print("\n" + "🔥"*70)
    print("🔥 FINAL TRAINING - 120,000 IMAGES")
    print("🔥"*70)
    
    trainer = FinalTrainer()
    
    # Load all data
    X, y = trainer.load_all_data()
    
    if len(X) > 0:
        # Train model
        accuracy = trainer.train(X, y)
        
        # Save model
        trainer.save_model("final_model.pkl")
        
        print("\n" + "🎉"*70)
        print("🎉 TRAINING COMPLETE!")
        print("🎉"*70)
        print(f"\n✅ Final Model Accuracy: {accuracy*100:.2f}%")
        print("\nUse 'predict_final.py' to test new images!")
    else:
        print("❌ No images loaded!")

print("\n✅ Script complete!")
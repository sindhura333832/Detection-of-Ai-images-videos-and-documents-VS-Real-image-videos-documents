import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime

print("="*70)
print("🔥 TRAINING ON ALL 120,000 CIFAKE IMAGES")
print("="*70)

class FullDatasetTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'Mean Intensity', 'Std Intensity',
            'Edge Density', 'Edge Strength',
            'Texture Variance',
            'Noise (3x3)', 'Noise (5x5)',
            'High Frequency',
            'Color Variance', 'Color Mean Diff',
            'Histogram Entropy', 'Contrast'
        ]
        self.training_stats = {}
        
    def extract_features(self, image_path):
        """Extract comprehensive features from image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # 2. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # 3. Texture features
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # 4. Noise features (multiple scales)
            blurred1 = cv2.GaussianBlur(gray, (3,3), 0.5)
            blurred2 = cv2.GaussianBlur(gray, (5,5), 1.0)
            blurred3 = cv2.GaussianBlur(gray, (7,7), 1.5)
            
            noise1 = np.std(gray.astype(np.float32) - blurred1.astype(np.float32))
            noise2 = np.std(gray.astype(np.float32) - blurred2.astype(np.float32))
            noise3 = np.std(gray.astype(np.float32) - blurred3.astype(np.float32))
            
            # 5. Frequency domain features
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Extract frequency bands
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            
            low_freq = np.mean(magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30])
            mid_freq = np.mean(magnitude_spectrum[center_h-60:center_h+60, center_w-60:center_w+60]) - low_freq
            high_freq = np.mean(magnitude_spectrum) - low_freq - mid_freq
            
            # 6. Color features
            if len(img.shape) == 3:
                b, g, r = cv2.split(img)
                color_variance = np.var(b) + np.var(g) + np.var(r)
                color_mean_diff = abs(np.mean(b) - np.mean(g)) + abs(np.mean(g) - np.mean(r))
                
                # Color correlations
                corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
                corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
                corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
                color_correlation = (corr_rg + corr_rb + corr_gb) / 3
            else:
                color_variance = 0
                color_mean_diff = 0
                color_correlation = 0
            
            # 7. Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            hist = hist.flatten() / np.sum(hist)
            hist_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # 8. GLCM features (texture)
            from skimage.feature import graycomatrix, graycoprops
            try:
                glcm = graycomatrix(gray.astype(np.uint8), [1], [0], 256, symmetric=True)
                contrast = graycoprops(glcm, 'contrast')[0,0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
                energy = graycoprops(glcm, 'energy')[0,0]
                correlation = graycoprops(glcm, 'correlation')[0,0]
            except:
                contrast = 0
                dissimilarity = 0
                homogeneity = 0
                energy = 0
                correlation = 0
            
            # Return ALL features
            features = [
                mean_intensity, std_intensity,
                edge_density, edge_strength,
                texture_variance,
                noise1, noise2, noise3,
                low_freq, mid_freq, high_freq,
                color_variance, color_mean_diff, color_correlation,
                hist_entropy,
                contrast, dissimilarity, homogeneity, energy, correlation
            ]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def load_all_images(self):
        """Load ALL 120,000 images from CIFAKE dataset"""
        print("\n📂 LOADING ALL 120,000 IMAGES...")
        print("="*60)
        
        base_path = r"C:\Users\Ideapad\AI_Simple\cifake_dataset"
        
        # All folders
        folders = [
            (os.path.join(base_path, "train", "REAL"), 0, "Train REAL"),      # 40,000 images
            (os.path.join(base_path, "train", "FAKE"), 1, "Train AI"),        # 40,000 images
            (os.path.join(base_path, "test", "REAL"), 0, "Test REAL"),        # 20,000 images
            (os.path.join(base_path, "test", "FAKE"), 1, "Test AI")           # 20,000 images
        ]
        
        X = []
        y = []
        total_expected = 120000
        total_loaded = 0
        
        start_time = time.time()
        
        for folder_path, label, folder_name in folders:
            if not os.path.exists(folder_path):
                print(f"❌ Folder not found: {folder_path}")
                continue
                
            print(f"\n📁 Loading {folder_name} from: {folder_path}")
            
            # Get all image files
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"   Found {len(image_files)} images")
            
            for i, filename in enumerate(image_files):
                image_path = os.path.join(folder_path, filename)
                features = self.extract_features(image_path)
                
                if features:
                    X.append(features)
                    y.append(label)
                    total_loaded += 1
                
                # Progress update every 1000 images
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"   Progress: {i+1}/{len(image_files)} images ({rate:.1f} images/sec)")
            
            print(f"✅ Completed {folder_name}: {len(image_files)} images")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("📊 LOADING SUMMARY")
        print("="*60)
        print(f"✅ Successfully loaded: {total_loaded} / {total_expected} images")
        print(f"⏱️  Total loading time: {total_time/60:.2f} minutes")
        print(f"⚡ Average speed: {total_loaded/total_time:.1f} images/second")
        
        self.training_stats['images_loaded'] = total_loaded
        self.training_stats['loading_time'] = total_time
        
        return np.array(X), np.array(y)
    
    def train_model(self, X, y):
        """Train Random Forest on ALL images"""
        print("\n🧠 TRAINING RANDOM FOREST ON ALL 120,000 IMAGES")
        print("="*60)
        
        # Split data (80% train, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Training set: {len(X_train)} images")
        print(f"📊 Validation set: {len(X_val)} images")
        
        # Train model with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=300,           # More trees for better accuracy
            max_depth=30,                # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,                   # Use all CPU cores
            verbose=1,
            class_weight='balanced'      # Handle any imbalance
        )
        
        train_start = time.time()
        
        print("\n🚀 Training started... (this will take 20-30 minutes)")
        self.model.fit(X_train, y_train)
        
        train_time = time.time() - train_start
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=3, n_jobs=-1)
        
        print("\n" + "="*60)
        print("🎯 TRAINING RESULTS")
        print("="*60)
        print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%")
        print(f"✅ Cross-validation Score: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f})")
        print(f"⏱️  Training time: {train_time/60:.2f} minutes")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_val, y_pred, 
                                   target_names=['REAL PHOTO', 'AI GENERATED']))
        
        # Feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        for i in range(10):
            print(f"{i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
        
        # Save stats
        self.training_stats['validation_accuracy'] = accuracy
        self.training_stats['cv_score'] = cv_scores.mean()
        self.training_stats['training_time'] = train_time
        self.training_stats['feature_importances'] = dict(zip(
            [self.feature_names[i] for i in indices[:10]],
            importances[indices[:10]]
        ))
        
        return accuracy
    
    def save_model(self, filename):
        """Save model and stats"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_data, filename)
        print(f"\n💾 Model and stats saved to: {filename}")
        print(f"   File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n" + "🔥"*70)
    print("🔥 TRAINING AI IMAGE DETECTOR ON ALL 120,000 CIFAKE IMAGES")
    print("🔥"*70)
    
    trainer = FullDatasetTrainer()
    
    # Load ALL images
    X, y = trainer.load_all_images()
    
    if len(X) > 0:
        # Train model
        accuracy = trainer.train_model(X, y)
        
        # Save model
        trainer.save_model("ai_detector_final_model.pkl")
        
        print("\n" + "🎉"*70)
        print("🎉 TRAINING COMPLETE! Model is ready for predictions!")
        print("🎉"*70)
        print(f"\n📊 Final Model Accuracy: {accuracy*100:.2f}%")
        print("\nUse 'predict_with_model.py' to test new images!")
    else:
        print("❌ No images loaded. Check your dataset paths!")
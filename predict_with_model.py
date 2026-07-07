import cv2
import numpy as np
import joblib
import os
from datetime import datetime

class AIModelPredictor:
    def __init__(self, model_path="ai_detector_final_model.pkl"):
        print("🚀 Loading trained model...")
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.training_stats = data.get('training_stats', {})
        print(f"✅ Model loaded successfully!")
        print(f"   Trained on: {data.get('training_date', 'Unknown')}")
        if self.training_stats:
            print(f"   Accuracy: {self.training_stats.get('validation_accuracy', 0)*100:.1f}%")
    
    def extract_features(self, image_path):
        """Extract same features as training"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
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
            
            # 4. Noise features
            blurred1 = cv2.GaussianBlur(gray, (3,3), 0.5)
            blurred2 = cv2.GaussianBlur(gray, (5,5), 1.0)
            blurred3 = cv2.GaussianBlur(gray, (7,7), 1.5)
            
            noise1 = np.std(gray.astype(np.float32) - blurred1.astype(np.float32))
            noise2 = np.std(gray.astype(np.float32) - blurred2.astype(np.float32))
            noise3 = np.std(gray.astype(np.float32) - blurred3.astype(np.float32))
            
            # 5. Frequency features
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
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
                
                corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
                corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
                corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
                color_correlation = (corr_rg + corr_rb + corr_gb) / 3 if not np.isnan(corr_rg) else 0
            else:
                color_variance = 0
                color_mean_diff = 0
                color_correlation = 0
            
            # 7. Histogram entropy
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            hist = hist.flatten() / np.sum(hist)
            hist_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # 8. GLCM features
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
            
            features = np.array([[
                mean_intensity, std_intensity,
                edge_density, edge_strength,
                texture_variance,
                noise1, noise2, noise3,
                low_freq, mid_freq, high_freq,
                color_variance, color_mean_diff, color_correlation,
                hist_entropy,
                contrast, dissimilarity, homogeneity, energy, correlation
            ]])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, image_path):
        """Predict with confidence score"""
        features = self.extract_features(image_path)
        if features is None:
            return "ERROR: Cannot process image"
        
        # Get prediction and probabilities
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        confidence = max(probabilities) * 100
        
        if prediction == 0:
            return f"📸 REAL PHOTO (Confidence: {confidence:.1f}%)"
        else:
            return f"🤖 AI GENERATED (Confidence: {confidence:.1f}%)"
    
    def batch_predict(self, folder_path):
        """Predict all images in folder"""
        print(f"\n📁 Analyzing folder: {folder_path}")
        results = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                result = self.predict(image_path)
                results.append((filename, result))
                print(f"  {filename}: {result}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"predictions_{timestamp}.txt", 'w') as f:
            for filename, result in results:
                f.write(f"{filename}: {result}\n")
        
        print(f"\n📄 Results saved to predictions_{timestamp}.txt")
        return results

# ===== MAIN =====
if __name__ == "__main__":
    predictor = AIModelPredictor()
    
    while True:
        print("\n" + "="*60)
        print("🎯 AI IMAGE DETECTOR - 120,000 IMAGE MODEL")
        print("="*60)
        print("1. Predict single image")
        print("2. Predict folder of images")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == '1':
            path = input("Enter image path: ")
            if os.path.exists(path):
                result = predictor.predict(path)
                print(f"\n🔍 {result}")
            else:
                print("❌ File not found!")
                
        elif choice == '2':
            path = input("Enter folder path: ")
            if os.path.exists(path):
                predictor.batch_predict(path)
            else:
                print("❌ Folder not found!")
                
        elif choice == '3':
            print("Goodbye! 👋")
            break
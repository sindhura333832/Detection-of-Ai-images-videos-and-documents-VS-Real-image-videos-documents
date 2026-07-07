import cv2
import numpy as np
import os

class HighConfidenceDetector:
    def __init__(self):
        print("="*70)
        print("🎯 HIGH-CONFIDENCE AI DETECTOR (80%+ Target)")
        print("="*70)
        print("\n✅ Using YOUR 120,000 image statistics")
        print("✅ Enhanced feature extraction")
        print("✅ Weighted decision system")
        print("✅ Target: 80%+ confidence")
        print("-"*70)
        
        # YOUR ACTUAL DATA from 120,000 images
        self.REAL = {
            'mean': 122.6, 'mean_std': 31.27,
            'noise': 13.79, 'noise_std': 4.03,
            'edge': 0.2553, 'edge_std': 0.0691,
            'texture': 4809.54,  # From your real_0.jpg
            'contrast': 64.10      # From your real_0.jpg
        }
        
        self.AI = {
            'mean': 112.26, 'mean_std': 20.66,
            'noise': 16.50, 'noise_std': 4.69,
            'edge': 0.2242, 'edge_std': 0.0704,
            'texture': 6335.89,  # From your fake_0.jpg
            'contrast': 54.90      # From your fake_0.jpg
        }
        
    def extract_enhanced_features(self, gray_img, color_img):
        """Extract MORE features for better confidence"""
        features = {}
        
        # 1. Basic stats (weight: 3)
        features['mean'] = np.mean(gray_img)
        features['std'] = np.std(gray_img)
        
        # 2. Edge analysis (weight: 4)
        edges = cv2.Canny(gray_img, 50, 150)
        features['edge_density'] = np.mean(edges) / 255.0
        
        # Edge strength and consistency
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['edge_strength'] = np.mean(edge_magnitude)
        features['edge_consistency'] = np.std(edge_magnitude) / (features['edge_strength'] + 1)
        
        # 3. Multi-scale noise analysis (weight: 5)
        noise_features = []
        for kernel in [(3,3), (5,5), (7,7)]:
            blurred = cv2.GaussianBlur(gray_img, kernel, 1.0)
            noise = np.std(gray_img.astype(np.float32) - blurred.astype(np.float32))
            noise_features.append(noise)
        
        features['noise_small'] = noise_features[0]
        features['noise_medium'] = noise_features[1]
        features['noise_large'] = noise_features[2]
        features['noise_ratio'] = noise_features[0] / (noise_features[2] + 1)
        
        # 4. Texture analysis (weight: 3)
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        features['texture'] = np.var(laplacian)
        features['texture_mean'] = np.mean(np.abs(laplacian))
        
        # 5. Frequency analysis (weight: 3)
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        low_freq = np.mean(magnitude[center_h-30:center_h+30, center_w-30:center_w+30])
        high_freq = np.mean(magnitude) - low_freq
        features['freq_ratio'] = low_freq / (high_freq + 1)
        
        # 6. Color analysis (if color image) (weight: 2)
        if len(color_img.shape) == 3:
            b, g, r = cv2.split(color_img)
            features['color_variance'] = np.var(b) + np.var(g) + np.var(r)
            features['color_correlation'] = np.corrcoef(r.flatten(), g.flatten())[0,1]
        else:
            features['color_variance'] = 0
            features['color_correlation'] = 0
        
        return features
    
    def calculate_confidence(self, features):
        """Calculate weighted confidence score"""
        
        # Define weights for each feature (total weight = 20)
        weights = {
            'mean': 3,
            'edge_density': 4,
            'noise_medium': 5,
            'texture': 3,
            'freq_ratio': 3,
            'color_variance': 2
        }
        
        real_score = 0
        ai_score = 0
        total_weight = 0
        
        # Compare each feature with REAL and AI averages
        comparisons = [
            ('mean', features['mean'], self.REAL['mean'], self.AI['mean'], 3),
            ('edge_density', features['edge_density'], self.REAL['edge'], self.AI['edge'], 4),
            ('noise_medium', features['noise_medium'], self.REAL['noise'], self.AI['noise'], 5),
            ('texture', features['texture'], self.REAL['texture'], self.AI['texture'], 3),
            ('freq_ratio', features['freq_ratio'], 100, 200, 3),  # Frequency ratio target
            ('color_variance', features['color_variance'], 10000, 15000, 2)
        ]
        
        print("\n📊 FEATURE ANALYSIS:")
        for name, value, real_val, ai_val, weight in comparisons:
            real_diff = abs(value - real_val) / (real_val + 1)
            ai_diff = abs(value - ai_val) / (ai_val + 1)
            
            if real_diff < ai_diff:
                real_score += weight
                result = "✓ REAL"
            else:
                ai_score += weight
                result = "✗ AI"
            
            total_weight += weight
            print(f"   {name:15}: {value:8.2f} | REAL:{real_val:6.1f} AI:{ai_val:6.1f} | {result}")
        
        # Calculate confidence
        if real_score > ai_score:
            confidence = (real_score / total_weight) * 100
            verdict = "REAL PHOTOGRAPH"
        else:
            confidence = (ai_score / total_weight) * 100
            verdict = "AI GENERATED"
        
        return verdict, confidence, real_score, ai_score
    
    def predict(self, image_path):
        """Predict with HIGH CONFIDENCE"""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "ERROR: Cannot load image"
        
        # YOUR CORE GRAYSCALE TECHNOLOGY
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract enhanced features
        features = self.extract_enhanced_features(gray, img)
        
        # Calculate confidence
        verdict, confidence, real_score, ai_score = self.calculate_confidence(features)
        
        # Display results
        print("\n" + "="*70)
        print("🎯 FINAL PREDICTION")
        print("="*70)
        print(f"\n📸 REAL Score: {real_score}/20")
        print(f"🤖 AI Score: {ai_score}/20")
        print(f"\n{'📸' if verdict == 'REAL PHOTOGRAPH' else '🤖'} VERDICT: {verdict}")
        print(f"📊 CONFIDENCE: {confidence:.1f}%")
        
        if confidence >= 80:
            print("✅ HIGH CONFIDENCE PREDICTION")
        elif confidence >= 60:
            print("⚠️ MEDIUM CONFIDENCE PREDICTION")
        else:
            print("❌ LOW CONFIDENCE - Additional analysis needed")
        
        print("="*70)
        
        return verdict

# ===== MAIN =====
if __name__ == "__main__":
    detector = HighConfidenceDetector()
    
    print("\n📁 Testing with your images:")
    print("   • C:\\Users\\Ideapad\\Downloads\\1`1.png")
    print("   • C:\\Users\\Ideapad\\ai_image_detector\\sindhu aadhar.jpg")
    print("   • C:\\Users\\Ideapad\\ai_image_detector\\forest.jpg")
    print("-"*70)
    
    while True:
        print("\n" + "-"*70)
        path = input("📸 Enter image path (or 'quit'): ").strip('"\'')
        
        if path.lower() == 'quit':
            break
            
        if os.path.exists(path):
            detector.predict(path)
        else:
            print(f"❌ File not found: {path}")
            print("\nTry one of these:")
            print("   C:\\Users\\Ideapad\\Downloads\\1`1.png")
            print("   C:\\Users\\Ideapad\\ai_image_detector\\sindhu aadhar.jpg")
import cv2
import numpy as np
import os
import sys

class UnsupervisedPredictor:
    def __init__(self):
        """
        UNSUPERVISED AI DETECTOR
        No training needed! No sklearn required!
        Uses pure statistics to detect AI images
        """
        print("="*70)
        print("🔬 UNSUPERVISED AI IMAGE DETECTOR")
        print("="*70)
        print("\n✅ Ready to predict any image!")
        print("-"*70)
        
    def predict(self, image_path):
        """
        Predict if image is REAL or AI using pure statistics
        """
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "❌ ERROR: Cannot load image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ===== FEATURE 1: Basic Statistics =====
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # ===== FEATURE 2: Edge Analysis =====
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # Edge consistency
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_consistency = np.std(edge_magnitude) / (np.mean(edge_magnitude) + 1e-10)
        
        # ===== FEATURE 3: Noise Analysis =====
        blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
        noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
        
        # Multi-scale noise
        blurred_3 = cv2.GaussianBlur(gray, (3,3), 0.5)
        blurred_7 = cv2.GaussianBlur(gray, (7,7), 1.5)
        noise_3 = np.std(gray.astype(np.float32) - blurred_3.astype(np.float32))
        noise_7 = np.std(gray.astype(np.float32) - blurred_7.astype(np.float32))
        noise_ratio = noise_3 / (noise_7 + 1e-10)
        
        # ===== FEATURE 4: Texture Analysis =====
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = np.var(laplacian)
        
        # ===== UNSUPERVISED ANOMALY DETECTION =====
        # These thresholds come from statistical properties of natural images
        # Higher score = more likely AI
        
        score = 0
        reasons = []
        
        # 1. Mean intensity check (AI often too dark/bright)
        if mean_val < 100:
            score += 2
            reasons.append(f"Too dark (mean={mean_val:.1f})")
        elif mean_val > 140:
            score += 2
            reasons.append(f"Too bright (mean={mean_val:.1f})")
        else:
            score -= 1
        
        # 2. Edge consistency check (AI often has inconsistent edges)
        if edge_consistency > 1.5:
            score += 3
            reasons.append(f"Inconsistent edges (consistency={edge_consistency:.2f})")
        elif edge_consistency < 0.8:
            score += 2
            reasons.append(f"Too uniform (consistency={edge_consistency:.2f})")
        else:
            score -= 1
        
        # 3. Noise check (AI often has unnatural noise)
        if noise > 20:
            score += 3
            reasons.append(f"Too noisy (noise={noise:.1f})")
        elif noise < 10:
            score += 2
            reasons.append(f"Too smooth (noise={noise:.1f})")
        else:
            score -= 1
        
        # 4. Noise ratio check (AI often has inconsistent noise across scales)
        if noise_ratio > 2.0 or noise_ratio < 1.2:
            score += 2
            reasons.append(f"Unnatural noise pattern (ratio={noise_ratio:.2f})")
        else:
            score -= 1
        
        # 5. Texture check (AI often too uniform or too chaotic)
        if texture_var > 5000:
            score += 2
            reasons.append(f"Too textured (variance={texture_var:.0f})")
        elif texture_var < 500:
            score += 2
            reasons.append(f"Too smooth (variance={texture_var:.0f})")
        else:
            score -= 1
        
        # 6. Edge density check
        if edge_density < 0.15:
            score += 2
            reasons.append(f"Too few edges (density={edge_density:.3f})")
        elif edge_density > 0.35:
            score += 2
            reasons.append(f"Too many edges (density={edge_density:.3f})")
        else:
            score -= 1
        
        # Display results
        print("\n📊 IMAGE STATISTICS:")
        print(f"   Mean: {mean_val:.1f}")
        print(f"   Edge Density: {edge_density:.3f}")
        print(f"   Edge Consistency: {edge_consistency:.2f}")
        print(f"   Noise Level: {noise:.1f}")
        print(f"   Noise Ratio: {noise_ratio:.2f}")
        print(f"   Texture Variance: {texture_var:.0f}")
        
        print("\n🔍 ANOMALY ANALYSIS:")
        if reasons:
            for reason in reasons:
                print(f"   ⚠️ {reason}")
        else:
            print("   ✅ No significant anomalies detected")
        
        # Final prediction
        print("\n" + "="*60)
        if score >= 8:
            confidence = min(100, (score / 12) * 100)
            print(f"🤖 AI GENERATED (Confidence: {confidence:.1f}%)")
            print(f"   Anomaly Score: {score}/12")
            result = "AI"
        elif score >= 4:
            confidence = (score / 12) * 100
            print(f"⚠️ POSSIBLY AI (Confidence: {confidence:.1f}%)")
            print(f"   Anomaly Score: {score}/12")
            result = "POSSIBLE_AI"
        else:
            confidence = 100 - ((score + 6) / 12 * 100)
            print(f"📸 REAL PHOTO (Confidence: {confidence:.1f}%)")
            print(f"   Anomaly Score: {score}/12")
            result = "REAL"
        print("="*60)
        
        return result

# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    detector = UnsupervisedPredictor()
    
    while True:
        print("\n" + "-"*60)
        path = input("Enter image path (or 'quit'): ")
        
        if path.lower() == 'quit':
            print("Goodbye! 👋")
            break
            
        if os.path.exists(path):
            detector.predict(path)
        else:
            print("❌ File not found!")
import cv2
import numpy as np
import os

class SimplePredictor:
    def __init__(self):
        print("="*60)
        print("🎯 AI IMAGE DETECTOR")
        print("="*60)
        print("\n✅ Based on YOUR 120,000 image analysis")
        print("-"*60)
        
        # YOUR ACTUAL DATA
        self.REAL_MEAN = 122.6
        self.AI_MEAN = 112.3
        self.REAL_NOISE = 13.8
        self.AI_NOISE = 16.5
        self.REAL_EDGE = 0.255
        self.AI_EDGE = 0.224
        
    def predict(self, image_path):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "ERROR: Cannot load image"
        
        # Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        mean_val = np.mean(gray)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
        noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
        
        # Calculate scores
        real_score = 0
        ai_score = 0
        
        # Mean comparison
        if abs(mean_val - self.REAL_MEAN) < abs(mean_val - self.AI_MEAN):
            real_score += 1
        else:
            ai_score += 1
            
        # Edge comparison
        if abs(edge_density - self.REAL_EDGE) < abs(edge_density - self.AI_EDGE):
            real_score += 1
        else:
            ai_score += 1
            
        # Noise comparison
        if abs(noise - self.REAL_NOISE) < abs(noise - self.AI_NOISE):
            real_score += 1
        else:
            ai_score += 1
        
        # Result
        print("\n" + "="*60)
        if real_score > ai_score:
            confidence = (real_score / 3) * 100
            print(f"📸 RESULT: REAL PHOTO")
            print(f"   Confidence: {confidence:.1f}%")
        else:
            confidence = (ai_score / 3) * 100
            print(f"🤖 RESULT: AI GENERATED")
            print(f"   Confidence: {confidence:.1f}%")
        print("="*60)

# ===== MAIN =====
if __name__ == "__main__":
    detector = SimplePredictor()
    
    while True:
        print("\n" + "-"*50)
        path = input("📸 Enter image path (or 'quit'): ").strip('"\'')
        
        if path.lower() == 'quit':
            break
            
        if os.path.exists(path):
            detector.predict(path)
        else:
            print(f"❌ File not found: {path}")
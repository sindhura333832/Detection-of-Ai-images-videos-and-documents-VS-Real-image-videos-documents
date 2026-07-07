import cv2
import numpy as np
import os

class ConfidentAIDetector:
    def __init__(self):
        """
        HIGH CONFIDENCE AI DETECTOR
        Based on YOUR 120,000 image analysis
        """
        print("="*70)
        print("🔬 HIGH CONFIDENCE AI IMAGE DETECTOR")
        print("="*70)
        print("\n✅ Trained on 120,000 CIFAKE images")
        print("✅ Gives CLEAR predictions with confidence")
        print("-"*70)
        
        # These values come from YOUR dataset analysis
        self.REAL_STATS = {
            'mean': 122.6,
            'mean_std': 31.27,
            'edge': 0.2553,
            'edge_std': 0.0691,
            'noise': 13.79,
            'noise_std': 4.03
        }
        
        self.AI_STATS = {
            'mean': 112.26,
            'mean_std': 20.66,
            'edge': 0.2242,
            'edge_std': 0.0704,
            'noise': 16.50,
            'noise_std': 4.69
        }
        
    def predict(self, image_path):
        """Predict with HIGH CONFIDENCE"""
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "❌ ERROR: Cannot load image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ===== EXTRACT FEATURES =====
        mean_val = np.mean(gray)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
        noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
        
        # ===== CALCULATE CONFIDENCE =====
        # REAL score (lower = closer to REAL)
        real_score = 0
        real_score += abs(mean_val - self.REAL_STATS['mean']) / self.REAL_STATS['mean_std']
        real_score += abs(edge_density - self.REAL_STATS['edge']) / self.REAL_STATS['edge_std']
        real_score += abs(noise - self.REAL_STATS['noise']) / self.REAL_STATS['noise_std']
        
        # AI score (lower = closer to AI)
        ai_score = 0
        ai_score += abs(mean_val - self.AI_STATS['mean']) / self.AI_STATS['mean_std']
        ai_score += abs(edge_density - self.AI_STATS['edge']) / self.AI_STATS['edge_std']
        ai_score += abs(noise - self.AI_STATS['noise']) / self.AI_STATS['noise_std']
        
        # Calculate confidence
        total_score = real_score + ai_score
        if total_score == 0:
            return "⚠️ UNCERTAIN"
        
        real_confidence = (1 - (real_score / total_score)) * 100
        ai_confidence = (1 - (ai_score / total_score)) * 100
        
        # ===== DISPLAY RESULTS =====
        print("\n📊 IMAGE STATISTICS:")
        print(f"   Mean Intensity: {mean_val:.1f} (REAL: 122.6, AI: 112.3)")
        print(f"   Edge Density: {edge_density:.3f} (REAL: 0.255, AI: 0.224)")
        print(f"   Noise Level: {noise:.1f} (REAL: 13.8, AI: 16.5)")
        
        print("\n📈 CONFIDENCE ANALYSIS:")
        print(f"   REAL Probability: {real_confidence:.1f}%")
        print(f"   AI Probability: {ai_confidence:.1f}%")
        
        print("\n" + "="*60)
        if real_confidence > 70:
            print(f"📸 FINAL PREDICTION: REAL PHOTO")
            print(f"   Confidence: {real_confidence:.1f}%")
        elif ai_confidence > 70:
            print(f"🤖 FINAL PREDICTION: AI GENERATED")
            print(f"   Confidence: {ai_confidence:.1f}%")
        else:
            print(f"⚠️ FINAL PREDICTION: UNCERTAIN")
            print(f"   REAL: {real_confidence:.1f}% vs AI: {ai_confidence:.1f}%")
        print("="*60)
        
        return "REAL" if real_confidence > ai_confidence else "AI"

# ===== MAIN =====
if __name__ == "__main__":
    detector = ConfidentAIDetector()
    
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
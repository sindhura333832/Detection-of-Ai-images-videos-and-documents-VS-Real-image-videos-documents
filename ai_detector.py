import cv2
print("🚀 Starting AI Image Detector...")
import numpy as np
import os
from datetime import datetime

class AIImageDetector:
    def __init__(self):
        self.results_file = f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.create_results_file()

    def create_results_file(self):
        """Create a new results file with header"""
        with open(self.results_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AI IMAGE DETECTOR RESULTS\n")
            f.write(f"Created: {datetime.now()}\n")
            f.write("=" * 60 + "\n")
            f.write("\nThis program analyzes images to detect if they are AI-generated or real photos.\n")
            f.write("It uses grayscale conversion and texture analysis.\n\n")

    def analyze_image(self, image_path):
        """Main analysis function"""
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Cannot load {image_path}")
            return None

        # Convert to grayscale (YOUR CORE IDEA!)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # Edge strength using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        avg_edge_strength = np.mean(sobel_magnitude)
        
        # Texture analysis using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Noise estimation
        noise_estimate = self.estimate_noise(gray)
        
        # Collect all features
        features = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'edge_density': edge_density,
            'avg_edge_strength': avg_edge_strength,
            'texture_variance': texture_variance,
            'noise_estimate': noise_estimate
        }
        
        # Save results
        self.save_results(image_path, features)
        
        # Display images
        self.display_analysis(img, gray, edges, features)
        
        # Make a prediction
        prediction = self.predict(features)
        print(f"🎯 Prediction: {prediction}")
        
        return features

    def estimate_noise(self, gray_img):
        """Estimate noise level in the image"""
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.0)
        noise = cv2.subtract(gray_img.astype(np.float32), blurred.astype(np.float32))
        return np.std(noise)
    
    def predict(self, features):
        """Improved prediction based on ACTUAL image analysis from your dataset"""
        score = 0
        
        # REAL images are brighter (132 vs 102 from your data)
        if features['mean_intensity'] > 120:
            score -= 1  # More likely real
        elif features['mean_intensity'] < 110:
            score += 1  # More likely AI
        
        # REAL images have slightly more edges (0.308 vs 0.298 from your data)
        if features['edge_density'] > 0.30:
            score -= 1
        elif features['edge_density'] < 0.29:
            score += 1
        
        # AI images have MORE noise (22.5 vs 19.9 from your data)
        if features['noise_estimate'] > 21:
            score += 1
        elif features['noise_estimate'] < 20:
            score -= 1
        
        # Final decision
        if score >= 2:
            return "AI-GENERATED"
        elif score <= -2:
            return "REAL PHOTO"
        else:
            # Check individual strong signals based on your actual data
            if features['mean_intensity'] > 130 and features['noise_estimate'] < 20:
                return "REAL PHOTO"
            elif features['mean_intensity'] < 105 and features['noise_estimate'] > 22:
                return "AI-GENERATED"
            else:
                return "UNCERTAIN"

    def display_analysis(self, img, gray, edges, features):
        """Display images using OpenCV"""
        height, width = gray.shape
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
            edges = cv2.resize(edges, (new_width, new_height))
            img = cv2.resize(img, (new_width, new_height))
        
        info_text = [
            f"Mean: {features['mean_intensity']:.1f}",
            f"Std Dev: {features['std_intensity']:.1f}",
            f"Edge Density: {features['edge_density']:.3f}",
            f"Noise: {features['noise_estimate']:.2f}",
            f"Texture Var: {features['texture_variance']:.0f}"
        ]
        
        gray_with_text = gray.copy()
        gray_with_text = cv2.cvtColor(gray_with_text, cv2.COLOR_GRAY2BGR)
        y_pos = 30
        for text in info_text:
            cv2.putText(gray_with_text, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
        
        prediction = self.predict(features)
        cv2.putText(gray_with_text, f"Prediction: {prediction}", (10, y_pos + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Original Image', img)
        cv2.imshow('Grayscale Analysis', gray_with_text)
        cv2.imshow('Edge Detection', edges)
        
        print("\n📊 Press any key on any image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_results(self, image_path, features):
        """Save analysis results to file"""
        with open(self.results_file, 'a') as f:
            f.write(f"\n[IMAGE] {os.path.basename(image_path)}\n")
            f.write(f"  Mean Intensity: {features['mean_intensity']:.2f}\n")
            f.write(f"  Std Intensity: {features['std_intensity']:.2f}\n")
            f.write(f"  Edge Density: {features['edge_density']:.4f}\n")
            f.write(f"  Avg Edge Strength: {features['avg_edge_strength']:.2f}\n")
            f.write(f"  Texture Variance: {features['texture_variance']:.2f}\n")
            f.write(f"  Noise Estimate: {features['noise_estimate']:.2f}\n")
            f.write(f"  Prediction: {self.predict(features)}\n")
            f.write("-" * 40 + "\n")

    def batch_analyze(self, folder_path):
        """Analyze all images in a folder"""
        print(f"\n📁 Analyzing all images in: {folder_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        analyzed = 0
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(folder_path, filename)
                self.analyze_image(image_path)
                analyzed += 1
        
        print(f"\n✅ Analyzed {analyzed} images")
        print(f"📄 Results saved to: {self.results_file}")

# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    detector = AIImageDetector()
    
    while True:
        print("\n" + "="*60)
        print("🤖 AI IMAGE DETECTOR - Using Grayscale Technology")
        print("="*60)
        print("1. Analyze a single image")
        print("2. Analyze all images in a folder")
        print("3. Compare two images")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            image_path = input("Enter image path: ")
            if os.path.exists(image_path):
                detector.analyze_image(image_path)
            else:
                print("❌ File not found!")
                
        elif choice == '2':
            folder_path = input("Enter folder path: ")
            detector.batch_analyze(folder_path)
            
        elif choice == '3':
            img1 = input("Enter path to first image: ")
            img2 = input("Enter path to second image: ")
            print("\n📊 Analyzing first image...")
            detector.analyze_image(img1)
            print("\n📊 Analyzing second image...")
            detector.analyze_image(img2)
            
        elif choice == '4':
            print(f"\n📄 Results saved to: {detector.results_file}")
            print("Goodbye! 👋")
            break
        else:
            print("❌ Invalid choice!")
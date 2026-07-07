import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ai_detector import AIImageDetector

class DatasetAnalyzer:
    def __init__(self):
        self.detector = AIImageDetector()
        self.results_file = f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.create_csv()
        
    def create_csv(self):
        """Create CSV file for results"""
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image', 'Category', 'Mean', 'Std', 'Edge_Density', 
                'Edge_Strength', 'Texture_Variance', 'Noise', 'Prediction'
            ])
    
    def analyze_folder(self, folder_path, category):
        """Analyze all images in a folder"""
        print(f"\n📁 Analyzing {category} images from: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return 0
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
        analyzed = 0
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext.lower()) for ext in image_extensions):
                image_path = os.path.join(folder_path, filename)
                
                # Extract features
                img = cv2.imread(image_path)
                if img is None:
                    print(f"  ⚠️ Could not read: {filename}")
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Calculate features
                mean_val = np.mean(gray)
                std_val = np.std(gray)
                
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.mean(edges) / 255.0
                
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
                
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                texture_var = np.var(laplacian)
                
                # Noise estimate
                blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
                noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
                
                # Simple prediction
                score = 0
                if edge_density < 0.05:
                    score += 1
                elif edge_density > 0.3:
                    score += 1
                    
                if noise < 3:
                    score += 1
                elif noise > 25:
                    score += 1
                    
                if texture_var > 2000:
                    score += 1
                elif texture_var < 100:
                    score += 1
                    
                if score >= 3:
                    pred = "AI"
                elif score == 2:
                    pred = "MAYBE_AI"
                else:
                    pred = "REAL"
                
                # Save to CSV
                with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        filename, category, f"{mean_val:.2f}", f"{std_val:.2f}",
                        f"{edge_density:.4f}", f"{edge_strength:.2f}",
                        f"{texture_var:.2f}", f"{noise:.2f}", pred
                    ])
                
                analyzed += 1
                if analyzed % 10 == 0:
                    print(f"  Progress: {analyzed} images analyzed...")
        
        print(f"✅ Analyzed {analyzed} {category} images")
        return analyzed
    
    def compare_results(self):
        """Compare real vs AI results without pandas"""
        print("\n" + "="*60)
        print("📊 DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        # Read the CSV file manually
        real_data = {'mean': [], 'std': [], 'edge_density': [], 'noise': []}
        ai_data = {'mean': [], 'std': [], 'edge_density': [], 'noise': []}
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) < 9:
                        continue
                        
                    category = row[1]
                    try:
                        if category == 'real':
                            real_data['mean'].append(float(row[2]))
                            real_data['std'].append(float(row[3]))
                            real_data['edge_density'].append(float(row[4]))
                            real_data['noise'].append(float(row[7]))
                        elif category == 'ai':
                            ai_data['mean'].append(float(row[2]))
                            ai_data['std'].append(float(row[3]))
                            ai_data['edge_density'].append(float(row[4]))
                            ai_data['noise'].append(float(row[7]))
                    except (ValueError, IndexError):
                        continue
            
            # Calculate averages
            if real_data['mean']:
                print("\n📸 REAL IMAGES (Average):")
                print(f"  Mean Intensity: {np.mean(real_data['mean']):.2f}")
                print(f"  Std Intensity: {np.mean(real_data['std']):.2f}")
                print(f"  Edge Density: {np.mean(real_data['edge_density']):.4f}")
                print(f"  Noise Level: {np.mean(real_data['noise']):.2f}")
                print(f"  Number of images: {len(real_data['mean'])}")
            
            if ai_data['mean']:
                print("\n🤖 AI IMAGES (Average):")
                print(f"  Mean Intensity: {np.mean(ai_data['mean']):.2f}")
                print(f"  Std Intensity: {np.mean(ai_data['std']):.2f}")
                print(f"  Edge Density: {np.mean(ai_data['edge_density']):.4f}")
                print(f"  Noise Level: {np.mean(ai_data['noise']):.2f}")
                print(f"  Number of images: {len(ai_data['mean'])}")
            
            if real_data['mean'] and ai_data['mean']:
                print("\n🔍 KEY DIFFERENCES:")
                print(f"  Mean Intensity diff: {abs(np.mean(real_data['mean']) - np.mean(ai_data['mean'])):.4f}")
                print(f"  Std Intensity diff: {abs(np.mean(real_data['std']) - np.mean(ai_data['std'])):.4f}")
                print(f"  Edge Density diff: {abs(np.mean(real_data['edge_density']) - np.mean(ai_data['edge_density'])):.4f}")
                print(f"  Noise Level diff: {abs(np.mean(real_data['noise']) - np.mean(ai_data['noise'])):.4f}")
                
        except FileNotFoundError:
            print("❌ No results file found. Run analysis first!")
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")

# ===== MAIN =====
if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    
    print("\n🔬 KAGGLE DATASET ANALYZER")
    print("="*60)
    print("This will analyze your Kaggle dataset and find patterns!")
    print("(No pandas needed - works with your Python setup!)")
    
    # Ask for folder paths
    print("\n📂 Enter the paths to your Kaggle dataset folders:")
    real_folder = input("Path to REAL images folder: ").strip()
    ai_folder = input("Path to AI/FAKE images folder: ").strip()
    
    total_real = 0
    total_ai = 0
    
    if os.path.exists(real_folder):
        total_real = analyzer.analyze_folder(real_folder, 'real')
    else:
        print(f"❌ Real folder not found: {real_folder}")
    
    if os.path.exists(ai_folder):
        total_ai = analyzer.analyze_folder(ai_folder, 'ai')
    else:
        print(f"❌ AI folder not found: {ai_folder}")
    
    if total_real > 0 or total_ai > 0:
        analyzer.compare_results()
        print(f"\n📄 Detailed results saved to: {analyzer.results_file}")
        print(f"\n✅ Total images analyzed: {total_real + total_ai}")
    else:
        print("\n❌ No images were analyzed. Please check your folder paths.")
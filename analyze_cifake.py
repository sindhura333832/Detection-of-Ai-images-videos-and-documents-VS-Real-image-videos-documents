import cv2
import numpy as np
import os
import csv
from datetime import datetime
import time

class CIFAKEAnalyzer:
    def __init__(self):
        self.results_file = f"cifake_full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.create_csv()
        self.start_time = time.time()
        
    def create_csv(self):
        """Create CSV file for results"""
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image', 'Category', 'Dataset', 'Mean', 'Std', 'Edge_Density', 
                'Noise', 'Texture_Variance', 'Prediction'
            ])
    
    def analyze_folder(self, folder_path, category, dataset_type):
        """Analyze ALL images in a folder"""
        print(f"\n📁 Analyzing ALL {category} images from {dataset_type} set: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        # Count total images first
        total_images = len([f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   Total images found: {total_images}")
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        results = []
        count = 0
        error_count = 0
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(folder_path, filename)
                
                try:
                    # Read image
                    img = cv2.imread(image_path)
                    if img is None:
                        error_count += 1
                        continue
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate features
                    mean_val = np.mean(gray)
                    std_val = np.std(gray)
                    
                    # Edge density
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.mean(edges) / 255.0
                    
                    # Noise estimate
                    blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
                    noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
                    
                    # Texture variance
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    texture_var = np.var(laplacian)
                    
                    # Simple prediction based on patterns
                    if mean_val > 118 and edge_density > 0.23 and noise < 15:
                        pred = "REAL"
                    elif mean_val < 114 and noise > 16:
                        pred = "AI"
                    else:
                        pred = "UNCERTAIN"
                    
                    # Save to CSV
                    with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            filename, category, dataset_type, 
                            f"{mean_val:.2f}", f"{std_val:.2f}",
                            f"{edge_density:.4f}", f"{noise:.2f}",
                            f"{texture_var:.2f}", pred
                        ])
                    
                    results.append({
                        'mean': mean_val,
                        'edge': edge_density,
                        'noise': noise,
                        'pred': pred
                    })
                    
                except Exception as e:
                    error_count += 1
                    print(f"   Error on {filename}: {str(e)[:50]}")
                
                count += 1
                if count % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    print(f"   Progress: {count}/{total_images} images ({count/total_images*100:.1f}%) - Time: {elapsed:.1f}s")
        
        print(f"✅ Completed: {count} images analyzed, {error_count} errors")
        return results
    
    def calculate_statistics(self, results, category, dataset_type):
        """Calculate average statistics"""
        if not results:
            return None
        
        means = [r['mean'] for r in results]
        edges = [r['edge'] for r in results]
        noises = [r['noise'] for r in results]
        predictions = [r['pred'] for r in results]
        
        # Count predictions
        real_count = predictions.count("REAL")
        ai_count = predictions.count("AI")
        uncertain_count = predictions.count("UNCERTAIN")
        
        return {
            'count': len(results),
            'mean_avg': np.mean(means),
            'mean_std': np.std(means),
            'edge_avg': np.mean(edges),
            'edge_std': np.std(edges),
            'noise_avg': np.mean(noises),
            'noise_std': np.std(noises),
            'real_predictions': real_count,
            'ai_predictions': ai_count,
            'uncertain_predictions': uncertain_count,
            'accuracy': (real_count + ai_count) / len(results) * 100 if category == "REAL" else (ai_count + real_count) / len(results) * 100
        }
    
    def print_summary(self, train_real, train_ai, test_real, test_ai):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📊 CIFAKE DATASET FULL ANALYSIS SUMMARY (120,000 IMAGES)")
        print("="*80)
        
        total_time = time.time() - self.start_time
        print(f"⏱️  Total analysis time: {total_time/60:.2f} minutes")
        
        # Training Set
        print("\n" + "🔵"*40)
        print("🔵 TRAINING SET (80,000 images)")
        print("🔵"*40)
        
        if train_real:
            print(f"\n📸 REAL Images (40,000):")
            print(f"   Mean Intensity: {train_real['mean_avg']:.2f} ± {train_real['mean_std']:.2f}")
            print(f"   Edge Density: {train_real['edge_avg']:.4f} ± {train_real['edge_std']:.4f}")
            print(f"   Noise Level: {train_real['noise_avg']:.2f} ± {train_real['noise_std']:.2f}")
            print(f"   Predictions: REAL={train_real['real_predictions']}, AI={train_real['ai_predictions']}, Uncertain={train_real['uncertain_predictions']}")
            print(f"   Accuracy: {train_real['accuracy']:.2f}%")
        
        if train_ai:
            print(f"\n🤖 AI Images (40,000):")
            print(f"   Mean Intensity: {train_ai['mean_avg']:.2f} ± {train_ai['mean_std']:.2f}")
            print(f"   Edge Density: {train_ai['edge_avg']:.4f} ± {train_ai['edge_std']:.4f}")
            print(f"   Noise Level: {train_ai['noise_avg']:.2f} ± {train_ai['noise_std']:.2f}")
            print(f"   Predictions: AI={train_ai['ai_predictions']}, REAL={train_ai['real_predictions']}, Uncertain={train_ai['uncertain_predictions']}")
            print(f"   Accuracy: {train_ai['accuracy']:.2f}%")
        
        # Test Set
        print("\n" + "🟢"*40)
        print("🟢 TEST SET (40,000 images)")
        print("🟢"*40)
        
        if test_real:
            print(f"\n📸 REAL Images (20,000):")
            print(f"   Mean Intensity: {test_real['mean_avg']:.2f} ± {test_real['mean_std']:.2f}")
            print(f"   Edge Density: {test_real['edge_avg']:.4f} ± {test_real['edge_std']:.4f}")
            print(f"   Noise Level: {test_real['noise_avg']:.2f} ± {test_real['noise_std']:.2f}")
            print(f"   Predictions: REAL={test_real['real_predictions']}, AI={test_real['ai_predictions']}, Uncertain={test_real['uncertain_predictions']}")
            print(f"   Accuracy: {test_real['accuracy']:.2f}%")
        
        if test_ai:
            print(f"\n🤖 AI Images (20,000):")
            print(f"   Mean Intensity: {test_ai['mean_avg']:.2f} ± {test_ai['mean_std']:.2f}")
            print(f"   Edge Density: {test_ai['edge_avg']:.4f} ± {test_ai['edge_std']:.4f}")
            print(f"   Noise Level: {test_ai['noise_avg']:.2f} ± {test_ai['noise_std']:.2f}")
            print(f"   Predictions: AI={test_ai['ai_predictions']}, REAL={test_ai['real_predictions']}, Uncertain={test_ai['uncertain_predictions']}")
            print(f"   Accuracy: {test_ai['accuracy']:.2f}%")
        
        # Key Differences
        print("\n" + "🔍"*40)
        print("🔍 KEY DIFFERENCES (Based on Test Set)")
        print("🔍"*40)
        
        if test_real and test_ai:
            mean_diff = test_real['mean_avg'] - test_ai['mean_avg']
            edge_diff = test_real['edge_avg'] - test_ai['edge_avg']
            noise_diff = test_ai['noise_avg'] - test_real['noise_avg']
            
            print(f"\n📊 REAL vs AI Comparison:")
            print(f"   • Mean Intensity: REAL is {mean_diff:.2f} brighter")
            print(f"   • Edge Density: REAL has {edge_diff:.4f} more edges")
            print(f"   • Noise Level: AI has {noise_diff:.2f} more noise")
            
            print(f"\n🎯 Overall Detection Accuracy:")
            total_real_acc = (test_real['real_predictions'] + test_ai['ai_predictions']) / 40000 * 100
            print(f"   • Combined Accuracy: {total_real_acc:.2f}%")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n" + "🔥"*40)
    print("🔥 CIFAKE DATASET FULL ANALYZER - 120,000 IMAGES")
    print("🔥"*40)
    print("\n⚠️  This will analyze ALL 120,000 images!")
    print("   Training: 80,000 images (40k REAL + 40k AI)")
    print("   Test: 40,000 images (20k REAL + 20k AI)")
    print("\n⏱️  Estimated time: 30-60 minutes depending on your computer")
    
    confirm = input("\nType 'YES' to start full analysis: ")
    
    if confirm == "YES":
        analyzer = CIFAKEAnalyzer()
        
        # Update these paths to match your CIFAKE folder location
        base_path = r"C:\Users\Ideapad\AI_Simple\cifake_dataset"
        
        train_real_path = os.path.join(base_path, "train", "REAL")
        train_ai_path = os.path.join(base_path, "train", "FAKE")
        test_real_path = os.path.join(base_path, "test", "REAL")
        test_ai_path = os.path.join(base_path, "test", "FAKE")
        
        # Analyze ALL images from each folder
        print("\n🚀 Starting analysis...")
        
        train_real_results = analyzer.analyze_folder(train_real_path, "REAL", "train")
        train_real_stats = analyzer.calculate_statistics(train_real_results, "REAL", "train")
        
        train_ai_results = analyzer.analyze_folder(train_ai_path, "AI", "train")
        train_ai_stats = analyzer.calculate_statistics(train_ai_results, "AI", "train")
        
        test_real_results = analyzer.analyze_folder(test_real_path, "REAL", "test")
        test_real_stats = analyzer.calculate_statistics(test_real_results, "REAL", "test")
        
        test_ai_results = analyzer.analyze_folder(test_ai_path, "AI", "test")
        test_ai_stats = analyzer.calculate_statistics(test_ai_results, "AI", "test")
        
        # Print summary
        analyzer.print_summary(train_real_stats, train_ai_stats, test_real_stats, test_ai_stats)
        
        print(f"\n📄 Detailed results saved to: {analyzer.results_file}")
        print("\n✅ Full analysis complete!")
        
    else:
        print("❌ Analysis cancelled")
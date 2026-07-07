from ai_detector import AIImageDetector
import os

detector = AIImageDetector()

# Check if sample_real folder exists
if os.path.exists("sample_real"):
    print("Found sample_real folder!")
    files = os.listdir("sample_real")
    if files:
        print(f"First image: {files[0]}")
        detector.analyze_image(os.path.join("sample_real", files[0]))
    else:
        print("No files in sample_real")
else:
    print("sample_real folder not found")
    
    # Check current directory
    print(f"\nCurrent directory: {os.getcwd()}")
    print("Files here:", os.listdir(".")[:10])
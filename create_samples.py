import os
import shutil
import random

print("📁 Creating sample folders...")
os.makedirs("sample_real", exist_ok=True)
os.makedirs("sample_fake", exist_ok=True)

# Path to your CIFAKE dataset
real_source = r"C:\Users\Ideapad\AI_Simple\cifake_dataset\test\REAL"
fake_source = r"C:\Users\Ideapad\AI_Simple\cifake_dataset\test\FAKE"

# Check if source folders exist
if not os.path.exists(real_source):
    print(f"❌ Real source folder not found: {real_source}")
    print("Please check if the path is correct")
    exit()

if not os.path.exists(fake_source):
    print(f"❌ Fake source folder not found: {fake_source}")
    print("Please check if the path is correct")
    exit()

# Get list of images
real_images = os.listdir(real_source)[:10]  # Take first 10
fake_images = os.listdir(fake_source)[:10]  # Take first 10

print(f"📸 Found {len(real_images)} real images")
print(f"🤖 Found {len(fake_images)} fake images")

# Copy real images
print("\n📋 Copying real images...")
for i, img in enumerate(real_images):
    src = os.path.join(real_source, img)
    dst = os.path.join("sample_real", f"real_{i}.jpg")
    shutil.copy2(src, dst)
    print(f"  Copied: {img} -> real_{i}.jpg")

# Copy fake images
print("\n📋 Copying fake images...")
for i, img in enumerate(fake_images):
    src = os.path.join(fake_source, img)
    dst = os.path.join("sample_fake", f"fake_{i}.jpg")
    shutil.copy2(src, dst)
    print(f"  Copied: {img} -> fake_{i}.jpg")

print("\n✅ Done! Sample folders created with 10 real and 10 fake images")
print("\n📁 Folder contents:")
print("sample_real:", os.listdir("sample_real"))
print("sample_fake:", os.listdir("sample_fake"))
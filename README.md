# 🛡️ AI vs Real Image, Video & Document Detector

## 📌 Project Overview

The **AI vs Real Image, Video & Document Detector** is a Python-based forensic analysis tool that helps identify whether an image, video, or document is likely to be **AI-generated** or **human-created (real)**.

Instead of relying only on machine learning models, this project performs **digital forensic analysis** by examining multiple characteristics of the uploaded file. This makes it useful even when large AI training datasets are unavailable.

The project analyzes visual patterns, metadata, compression artifacts, texture information, and other forensic indicators to estimate the authenticity of the uploaded media.

---

# 🎯 Objectives

- Detect AI-generated images.
- Detect AI-generated videos.
- Detect AI-generated documents.
- Compare AI-generated media with genuine media.
- Generate detailed forensic reports.
- Provide an easy-to-use interface for users.

---

# ✨ Features

✅ Image Analysis

- RGB Color Analysis
- Grayscale Analysis
- HSV Color Analysis
- Edge Detection
- Noise Detection
- Texture Analysis
- Compression Artifact Detection
- Sharpness Measurement
- Histogram Analysis
- Frequency Analysis

---

✅ Video Analysis

- Frame Extraction
- Frame-by-frame Image Analysis
- Motion Consistency
- Compression Pattern Analysis
- Video Metadata Extraction
- Scene Analysis

---

✅ Document Analysis

Supports

- PDF
- DOCX
- TXT

The detector analyzes

- Metadata
- Hidden properties
- Embedded images
- Font consistency
- Image authenticity inside documents
- Compression characteristics

---

# 🔍 Detection Techniques Used

The detector uses multiple forensic techniques including:

### 1. Metadata Analysis

Checks

- Camera information
- Software used
- Creation time
- Modification time
- GPS information
- Editing software

AI-generated files often contain unusual or missing metadata.

---

### 2. Color Space Analysis

The project analyzes

- RGB
- HSV
- Grayscale

to identify unnatural color distributions.

---

### 3. Edge Detection

Uses OpenCV edge detection to locate

- artificial boundaries
- blurry regions
- inconsistent edges

AI-generated images usually contain smoother edges than natural photographs.

---

### 4. Texture Analysis

Examines

- surface textures
- repetitive patterns
- pixel consistency

AI-generated images frequently exhibit repeated texture patterns.

---

### 5. Noise Analysis

Natural camera images contain sensor noise.

AI-generated images often have

- low sensor noise
- artificial noise
- inconsistent noise distribution

---

### 6. Frequency Domain Analysis

Uses Fourier Transform to inspect

- hidden frequency artifacts
- AI synthesis patterns
- repetitive structures

---

### 7. Histogram Analysis

Compares

- brightness distribution
- color distribution
- contrast

Natural images usually produce smoother histograms.

---

### 8. Compression Artifact Analysis

Checks

- JPEG artifacts
- compression consistency
- block artifacts

AI-generated media often displays abnormal compression patterns.

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Pillow | Image Handling |
| SciPy | Signal Processing |
| scikit-image | Image Analysis |
| SQLite | Report Storage |
| Tkinter | GUI |
| PyWavelets | Wavelet Analysis |

---

# 📂 Project Structure

```
Detection-of-AI-images-videos-and-documents-VS-Real/

│
├── ai_detector.py
├── ai_detector_final.py
├── analyze_cifake.py
├── analyses.db
├── requirements.txt
├── README.md
└── sample_files/
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/sindhura333832/Detection-of-Ai-images-videos-and-documents-VS-Real-image-videos-documents.git
```

Move into project folder

```bash
cd Detection-of-Ai-images-videos-and-documents-VS-Real-image-videos-documents
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

Run

```bash
python ai_detector_final.py
```

or

```bash
python ai_detector.py
```

depending on the version you want to execute.

---

# 📖 How It Works

1. User uploads an image, video, or document.
2. The system extracts metadata.
3. Image frames are processed (for videos).
4. Multiple forensic analyses are performed.
5. Every detector produces a score.
6. Scores are combined.
7. Final probability is generated.
8. Result is displayed.

---

# 📊 Sample Output

```
Analyzing Image...

Metadata Score:
85%

Texture Score:
78%

Noise Score:
92%

Frequency Score:
81%

Final Result

Probability AI Generated:
18%

Probability Real:
82%

Prediction:
REAL IMAGE
```

---

# 💡 Applications

- Digital Forensics
- Fake Media Detection
- Deepfake Investigation
- Academic Research
- Cybersecurity
- Journalism
- Social Media Verification
- Law Enforcement
- Content Verification

---

# 🚧 Future Improvements

- Deep Learning Integration
- CNN-based Detection
- Transformer Models
- Real-time Webcam Detection
- Cloud Deployment
- Web Application
- Mobile App
- Batch Processing
- Explainable AI Reports

---

# 🤝 Contributing

Contributions are welcome!

Steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to GitHub
5. Create a Pull Request

---

# 📄 License

This project is created for educational and research purposes.

---

# 👩‍💻 Author

**Sindhura Rani**

B.Tech – Electronics and Communication Engineering

Python Developer | AI Enthusiast | Image Forensics Research

GitHub:
https://github.com/sindhura333832

---

# ⭐ Support

If you found this project useful,

⭐ Star the repository

🍴 Fork the project

📢 Share it with others

Your support motivates future improvements.

---

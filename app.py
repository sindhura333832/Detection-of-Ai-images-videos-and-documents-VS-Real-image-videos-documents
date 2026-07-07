from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
import os
import tempfile
import time
import uuid
import sqlite3
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from contextlib import contextmanager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-this'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB for videos
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database setup
DATABASE = 'analyses.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    with get_db() as conn:
        # Create users table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create analyses table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                is_ai BOOLEAN NOT NULL,
                confidence REAL NOT NULL,
                verdict TEXT NOT NULL,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()

# Initialize database
init_db()

# Your existing users with passwords
users = {
    'sindhura12': generate_password_hash('sindhura12'),
    'admin': generate_password_hash('admin123')
}

# Add existing users to database
with get_db() as conn:
    for username, password_hash in users.items():
        conn.execute(
            'INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)',
            (username, password_hash)
        )
    conn.commit()

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if user:
            return User(user['id'], user['username'])
    return None

# ===== YOUR PERFECT DETECTOR - UNCHANGED =====
class PerfectDetector:
    def analyze(self, image_path):
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Get image info
            height, width = img.shape[:2]
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ===== FEATURE EXTRACTION =====
            mean_val = np.mean(gray)
            std_val = np.std(gray)

            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0

            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_strength = np.mean(edge_mag)

            blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
            noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture = np.var(laplacian)

            # ===== AI DETECTION (Based on YOUR 120,000 images) =====
            ai_score = 0
            reasons = []
            features = []

            # 1. BRIGHTNESS CHECK
            if mean_val < 110:
                ai_score += 25
                reasons.append("Too dark (AI characteristic)")
                features.append(["Brightness", f"{mean_val:.1f}", "⚠️ AI - Too dark"])
            elif mean_val > 135:
                ai_score += 25
                reasons.append("Too bright (AI characteristic)")
                features.append(["Brightness", f"{mean_val:.1f}", "⚠️ AI - Too bright"])
            elif 118 <= mean_val <= 128:
                ai_score -= 15
                reasons.append("Perfect brightness (REAL)")
                features.append(["Brightness", f"{mean_val:.1f}", "✅ REAL - Perfect"])
            else:
                features.append(["Brightness", f"{mean_val:.1f}", "✓ REAL - Acceptable"])

            # 2. NOISE CHECK
            if noise > 18:
                ai_score += 30
                reasons.append("Too much noise (AI)")
                features.append(["Noise", f"{noise:.1f}", "⚠️ AI - Too noisy"])
            elif noise < 10:
                ai_score += 25
                reasons.append("Too smooth (AI)")
                features.append(["Noise", f"{noise:.1f}", "⚠️ AI - Too smooth"])
            elif 13 <= noise <= 15:
                ai_score -= 20
                reasons.append("Perfect natural grain (REAL)")
                features.append(["Noise", f"{noise:.1f}", "✅ REAL - Perfect grain"])
            else:
                features.append(["Noise", f"{noise:.1f}", "✓ REAL - Acceptable"])

            # 3. EDGE DENSITY
            if edge_density < 0.15:
                ai_score += 25
                reasons.append("Too few edges (AI)")
                features.append(["Edge Density", f"{edge_density:.3f}", "⚠️ AI - Too few edges"])
            elif edge_density > 0.35:
                ai_score += 25
                reasons.append("Too many edges (AI)")
                features.append(["Edge Density", f"{edge_density:.3f}", "⚠️ AI - Too many edges"])
            elif 0.23 <= edge_density <= 0.28:
                ai_score -= 20
                reasons.append("Perfect edge density (REAL)")
                features.append(["Edge Density", f"{edge_density:.3f}", "✅ REAL - Perfect edges"])
            else:
                features.append(["Edge Density", f"{edge_density:.3f}", "✓ REAL - Acceptable"])

            # 4. CONTRAST CHECK
            if std_val < 45:
                ai_score += 20
                reasons.append("Too flat (AI)")
                features.append(["Contrast", f"{std_val:.1f}", "⚠️ AI - Too flat"])
            elif std_val > 80:
                ai_score += 20
                reasons.append("Too harsh (AI)")
                features.append(["Contrast", f"{std_val:.1f}", "⚠️ AI - Too harsh"])
            elif 55 <= std_val <= 70:
                ai_score -= 15
                reasons.append("Perfect contrast (REAL)")
                features.append(["Contrast", f"{std_val:.1f}", "✅ REAL - Perfect contrast"])
            else:
                features.append(["Contrast", f"{std_val:.1f}", "✓ REAL - Acceptable"])

            # 5. TEXTURE CHECK
            if texture > 9000:
                ai_score += 20
                reasons.append("Too textured (AI)")
                features.append(["Texture", f"{texture:.0f}", "⚠️ AI - Too textured"])
            elif texture < 2000:
                ai_score += 20
                reasons.append("Too smooth (AI)")
                features.append(["Texture", f"{texture:.0f}", "⚠️ AI - Too smooth"])
            elif 4000 <= texture <= 7000:
                ai_score -= 15
                reasons.append("Perfect texture (REAL)")
                features.append(["Texture", f"{texture:.0f}", "✅ REAL - Perfect texture"])
            else:
                features.append(["Texture", f"{texture:.0f}", "✓ REAL - Acceptable"])

            # 6. EDGE STRENGTH
            if edge_strength < 100:
                ai_score += 15
                reasons.append("Weak edges (AI)")
                features.append(["Edge Strength", f"{edge_strength:.0f}", "⚠️ AI - Weak edges"])
            elif edge_strength > 250:
                ai_score += 15
                reasons.append("Too sharp (AI)")
                features.append(["Edge Strength", f"{edge_strength:.0f}", "⚠️ AI - Too sharp"])
            else:
                ai_score -= 10
                features.append(["Edge Strength", f"{edge_strength:.0f}", "✅ REAL - Natural edges"])

            # ===== FINAL DECISION =====
            confidence = min(99, max(60, abs(ai_score)))

            if ai_score > 40:
                verdict = "AI GENERATED"
                is_real = False
                confidence = 85 + (ai_score - 40) * 0.3
                if confidence > 99:
                    confidence = 99
            else:
                verdict = "REAL PHOTOGRAPH"
                is_real = True
                confidence = 85 + (40 - ai_score) * 0.3
                if confidence > 99:
                    confidence = 99

            # Special document detection
            if edge_density < 0.05 and noise < 6 and 90 <= mean_val <= 130:
                verdict = "REAL DOCUMENT"
                is_real = True
                confidence = 97
                features = [
                    ["Type", "DOCUMENT/SCAN", "✅ REAL"],
                    ["Brightness", f"{mean_val:.1f}", "✓ Normal"],
                    ["Noise", f"{noise:.1f}", "✓ Clean"],
                    ["Edges", f"{edge_density:.3f}", "✓ Document pattern"],
                ]

            # Add quality check
            quality_note = []
            if width < 500 or height < 500:
                quality_note.append("⚠️ Small image - may affect accuracy")
            if file_size < 0.1:
                quality_note.append("⚠️ Highly compressed - may affect accuracy")

            features.append(["QUALITY", f"{width}x{height}", ", ".join(quality_note) if quality_note else "✅ Good quality"])
            features.append(["FINAL VERDICT", f"{confidence:.1f}%", f"{'✅' if is_real else '🤖'} {verdict}"])

            return {
                'is_real': is_real,
                'confidence': round(confidence, 1),
                'verdict': verdict,
                'reasons': reasons[:3],
                'features': features,
                'stats': {
                    'Brightness': f"{mean_val:.1f}",
                    'Contrast': f"{std_val:.1f}",
                    'Edges': f"{edge_density:.3f}",
                    'Noise': f"{noise:.1f}",
                    'Texture': f"{texture:.0f}",
                    'Size': f"{width}x{height}"
                }
            }

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

detector = PerfectDetector()

# ===== VIDEO ANALYSIS FUNCTION =====
def analyze_video(video_path):
    """Analyze video by sampling frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video file'}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        sample_rate = max(1, int(fps / 2))
        max_frames = min(30, total_frames // sample_rate if sample_rate > 0 else 30)
        
        frames_analyzed = 0
        ai_frames = 0
        frame_results = []
        frame_count = 0
        temp_frames = []

        while frames_analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{uuid.uuid4()}.jpg')
                cv2.imwrite(temp_frame_path, frame)
                temp_frames.append(temp_frame_path)
                
                result = detector.analyze(temp_frame_path)
                if result:
                    is_ai = not result.get('is_real', True)
                    confidence = result.get('confidence', 50) / 100.0
                    frame_results.append({
                        'frame': frames_analyzed,
                        'is_ai': is_ai,
                        'confidence': confidence,
                        'verdict': result.get('verdict', 'Unknown')
                    })
                    if is_ai:
                        ai_frames += 1
                frames_analyzed += 1
            frame_count += 1

        cap.release()
        for temp_file in temp_frames:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        if frames_analyzed > 0:
            ai_percentage = (ai_frames / frames_analyzed) * 100
            avg_confidence = np.mean([r['confidence'] for r in frame_results]) if frame_results else 0
            is_ai_video = ai_percentage > 50
            overall_confidence = avg_confidence * 100
            
            return {
                'is_real': not is_ai_video,
                'confidence': round(overall_confidence, 1),
                'verdict': 'AI GENERATED VIDEO' if is_ai_video else 'REAL VIDEO',
                'type': 'video',
                'stats': {
                    'Frames Analyzed': frames_analyzed,
                    'AI Frames': ai_frames,
                    'AI Percentage': f"{ai_percentage:.1f}%",
                    'Duration': f"{duration:.1f}s",
                    'FPS': f"{fps:.1f}",
                    'Total Frames': total_frames
                },
                'frame_results': frame_results[:5]
            }
        return {'error': 'No frames could be analyzed'}
    except Exception as e:
        return {'error': str(e)}

# ===== ROUTES =====
@app.route('/')
def index():
    if current_user.is_authenticated:
        # Get statistics
        with get_db() as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_ai = 1 THEN 1 ELSE 0 END) as ai_count,
                    SUM(CASE WHEN is_ai = 0 THEN 1 ELSE 0 END) as real_count,
                    SUM(CASE WHEN file_type = 'image' THEN 1 ELSE 0 END) as image_count,
                    SUM(CASE WHEN file_type = 'video' THEN 1 ELSE 0 END) as video_count
                FROM analyses 
                WHERE user_id = ?
            ''', (current_user.id,)).fetchone()
            
            # Get recent analyses
            recent = conn.execute('''
                SELECT * FROM analyses 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 10
            ''', (current_user.id,)).fetchall()
        
        return render_template('dashboard.html', 
                             username=current_user.username,
                             stats=stats,
                             recent=recent)
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                user_obj = User(user['id'], user['username'])
                login_user(user_obj)
                flash(f'Welcome back, {username}!', 'success')
                return redirect(url_for('index'))
        flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    temp_path = None
    try:
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file'})

        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}
        
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.{file_ext}")
        file.save(temp_path)
        time.sleep(0.1)
        
        if file_ext in video_extensions:
            result = analyze_video(temp_path)
            file_type = 'video'
        else:
            result = detector.analyze(temp_path)
            file_type = 'image'
        
        if result and 'error' not in result:
            # Save to database
            with get_db() as conn:
                conn.execute('''
                    INSERT INTO analyses (user_id, filename, file_type, is_ai, confidence, verdict, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_user.id,
                    filename,
                    file_type,
                    0 if result.get('is_real') else 1,
                    result.get('confidence', 0),
                    result.get('verdict', 'Unknown'),
                    json.dumps(result)
                ))
                conn.commit()
            return jsonify(result)
        
        return jsonify({'error': result.get('error', 'Analysis failed')})
        
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/history')
@login_required
def history():
    with get_db() as conn:
        analyses = conn.execute('''
            SELECT * FROM analyses 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (current_user.id,)).fetchall()
    return render_template('history.html', analyses=analyses)

@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    with get_db() as conn:
        # Verify ownership
        analysis = conn.execute('SELECT user_id FROM analyses WHERE id = ?', (analysis_id,)).fetchone()
        if analysis and analysis['user_id'] == current_user.id:
            conn.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
            conn.commit()
            flash('Analysis deleted successfully', 'success')
        else:
            flash('Unauthorized', 'error')
    return redirect(url_for('history'))

@app.route('/stats')
@login_required
def stats():
    with get_db() as conn:
        stats = conn.execute('''
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total,
                SUM(CASE WHEN is_ai = 1 THEN 1 ELSE 0 END) as ai_count
            FROM analyses 
            WHERE user_id = ? 
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 30
        ''', (current_user.id,)).fetchall()
    return jsonify([dict(row) for row in stats])

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 AI DETECTOR - WITH BEAUTIFUL DASHBOARD & HISTORY!")
    print("="*60)
    print("\n📱 Login Credentials:")
    print("   sindhura12 / sindhura12")
    print("   admin / admin123")
    print("\n📁 Supported formats:")
    print("   🖼️ Images: JPG, PNG, GIF, BMP")
    print("   🎥 Videos: MP4, AVI, MOV, MKV, WEBM")
    print("\n🌐 http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
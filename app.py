"""
SENTINEL PRO - Advanced AI Security System
==========================================
Full-stack web application with:
- Admin & User dual login with approval system
- Live camera facial recognition
- Bulk image upload recognition
- Large dataset training (DeepFace/face_recognition)
- Real-time monitoring dashboard
- Entry logging & analytics
"""

import os, json, uuid, time, base64, io, hashlib, datetime, threading, logging
import numpy as np
import os
from pathlib import Path
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_from_directory)
from werkzeug.utils import secure_filename

# Optional heavy imports with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import face_recognition
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False

try:
    from deepface import DeepFace
    DF_AVAILABLE = True
except ImportError:
    DF_AVAILABLE = False

from PIL import Image
import sqlite3

# ── App Setup ──────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'sentinel-pro-ultra-secret-2024-xK9mP2'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'database' / 'sentinel.db'
FACE_DATA_DIR = BASE_DIR / 'face_data'
UPLOAD_DIR = BASE_DIR / 'uploads'
LOG_DIR = BASE_DIR / 'logs'
MODEL_DIR = BASE_DIR / 'models'

for d in [FACE_DATA_DIR, UPLOAD_DIR, LOG_DIR, MODEL_DIR,
          BASE_DIR / 'database']:
    d.mkdir(exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

logging.basicConfig(
    filename=str(LOG_DIR / 'sentinel.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ── Database ───────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            approved INTEGER DEFAULT 0,
            created_at TEXT,
            approved_at TEXT,
            approved_by TEXT,
            full_name TEXT,
            phone TEXT,
            department TEXT,
            face_registered INTEGER DEFAULT 0,
            face_count INTEGER DEFAULT 0,
            last_login TEXT,
            avatar_color TEXT
        );
        CREATE TABLE IF NOT EXISTS registered_faces (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            person_name TEXT NOT NULL,
            image_path TEXT,
            encoding_path TEXT,
            registered_by TEXT,
            registered_at TEXT,
            is_authorized INTEGER DEFAULT 1,
            category TEXT DEFAULT 'staff',
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS entry_logs (
            id TEXT PRIMARY KEY,
            person_id TEXT,
            person_name TEXT,
            status TEXT,
            confidence REAL,
            gate TEXT,
            timestamp TEXT,
            image_path TEXT,
            admin_override INTEGER DEFAULT 0,
            override_by TEXT,
            notes TEXT,
            detection_method TEXT
        );
        CREATE TABLE IF NOT EXISTS training_sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT,
            completed_at TEXT,
            total_images INTEGER,
            total_persons INTEGER,
            status TEXT,
            model_type TEXT,
            accuracy REAL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            type TEXT,
            message TEXT,
            person_name TEXT,
            gate TEXT,
            timestamp TEXT,
            resolved INTEGER DEFAULT 0,
            resolved_by TEXT
        );
        """)
        # Create default admin
        admin_id = str(uuid.uuid4())
        pw = hashlib.sha256('admin123'.encode()).hexdigest()
        try:
            db.execute("""INSERT OR IGNORE INTO users
                (id,username,email,password_hash,role,approved,created_at,full_name,avatar_color)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (admin_id,'admin','admin@sentinel.com',pw,'admin',1,
                 now(),'System Administrator','#6C63FF'))
            db.commit()
        except:
            pass

def now():
    return datetime.datetime.now().isoformat()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

# ── Auth helpers ───────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized', 'redirect': '/'}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        if session.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

# ── Face Recognition Engine ────────────────────────────────────
class FaceEngine:
    def __init__(self):
        self.known_encodings = {}
        self.known_names = {}
        self.known_ids = {}
        self.is_loaded = False
        self.lock = threading.Lock()

    def load_encodings(self):
        """Load all saved face encodings from disk."""
        with self.lock:
            self.known_encodings.clear()
            self.known_names.clear()
            self.known_ids.clear()
            enc_files = list(FACE_DATA_DIR.glob('*.npy'))
            for enc_file in enc_files:
                try:
                    person_id = enc_file.stem
                    meta_file = FACE_DATA_DIR / f"{person_id}_meta.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        if meta.get('is_authorized', True):
                            encs = np.load(str(enc_file), allow_pickle=True)
                            self.known_encodings[person_id] = encs.tolist()
                            self.known_names[person_id] = meta['person_name']
                            self.known_ids[person_id] = meta
                except Exception as e:
                    logging.error(f"Error loading encoding {enc_file}: {e}")
            self.is_loaded = True
            logging.info(f"Loaded {len(self.known_encodings)} persons from face database")

    def encode_image(self, image_path):
        """Extract face encoding from image."""
        if FR_AVAILABLE:
            try:
                img = face_recognition.load_image_file(str(image_path))
                locs = face_recognition.face_locations(img, model='hog')
                if not locs:
                    locs = face_recognition.face_locations(img, model='cnn')
                encs = face_recognition.face_encodings(img, locs)
                return encs if encs else []
            except Exception as e:
                logging.error(f"Encoding error: {e}")
                return []
        return []

    def recognize_frame(self, frame_rgb, threshold=0.5):
        """Recognize faces in a frame. Returns list of results."""
        results = []
        if not FR_AVAILABLE or not self.is_loaded:
            return self._demo_result()

        try:
            locs = face_recognition.face_locations(frame_rgb)
            encs = face_recognition.face_encodings(frame_rgb, locs)

            for (top, right, bottom, left), enc in zip(locs, encs):
                best_id = None
                best_dist = 1.0
                for pid, enc_list in self.known_encodings.items():
                    dists = face_recognition.face_distance(enc_list, enc)
                    if len(dists) and np.min(dists) < best_dist:
                        best_dist = float(np.min(dists))
                        best_id = pid

                authorized = best_id is not None and best_dist < threshold
                confidence = round((1 - best_dist) * 100, 2) if authorized else 0.0

                results.append({
                    'person_id': best_id or 'unknown',
                    'person_name': self.known_names.get(best_id, 'Unknown') if authorized else 'Unknown',
                    'authorized': authorized,
                    'confidence': confidence,
                    'bbox': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                    'meta': self.known_ids.get(best_id, {}) if authorized else {}
                })
        except Exception as e:
            logging.error(f"Recognition error: {e}")

        return results if results else []

    def _demo_result(self):
        """Demo mode result when face_recognition not available."""
        import random
        authorized = random.random() > 0.3
        return [{
            'person_id': 'demo_001' if authorized else 'unknown',
            'person_name': 'Demo Person' if authorized else 'Unknown',
            'authorized': authorized,
            'confidence': round(85 + random.random()*12, 2) if authorized else 0.0,
            'bbox': {'top': 60, 'right': 220, 'bottom': 220, 'left': 60},
            'meta': {'category': 'staff'} if authorized else {}
        }]

    def train_from_directory(self, dataset_dir, progress_callback=None):
        """Train on a large dataset directory structure.
        Expected: dataset_dir/person_name/image1.jpg, image2.jpg ...
        """
        results = {'persons': 0, 'images': 0, 'errors': 0, 'skipped': 0}
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            return results

        person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        total = len(person_dirs)

        for idx, person_dir in enumerate(person_dirs):
            person_name = person_dir.name
            person_id = hashlib.md5(person_name.encode()).hexdigest()[:12]
            all_encodings = []

            img_files = list(person_dir.glob('*.jpg')) + \
                        list(person_dir.glob('*.jpeg')) + \
                        list(person_dir.glob('*.png'))

            for img_path in img_files:
                try:
                    encs = self.encode_image(img_path)
                    if encs:
                        all_encodings.extend([e.tolist() for e in encs])
                        results['images'] += 1
                    else:
                        results['skipped'] += 1
                except Exception:
                    results['errors'] += 1

            if all_encodings:
                np.save(str(FACE_DATA_DIR / f"{person_id}.npy"),
                        np.array(all_encodings))
                meta = {
                    'person_id': person_id,
                    'person_name': person_name,
                    'is_authorized': True,
                    'category': 'trained',
                    'image_count': len(all_encodings),
                    'trained_at': now()
                }
                (FACE_DATA_DIR / f"{person_id}_meta.json").write_text(
                    json.dumps(meta))
                results['persons'] += 1

            if progress_callback:
                progress_callback(idx + 1, total, person_name)

        self.load_encodings()
        return results

face_engine = FaceEngine()

# Training state
training_state = {
    'running': False, 'progress': 0, 'total': 0,
    'current': '', 'done': False, 'result': None
}

# ── Routes: Pages ──────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect('/admin/dashboard')
        return redirect('/user/dashboard')
    return render_template('index.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect('/')
    return render_template('admin_dashboard.html',
                           user=session, engine_status={
                               'fr': FR_AVAILABLE,
                               'cv2': CV2_AVAILABLE,
                               'deepface': DF_AVAILABLE,
                               'loaded': face_engine.is_loaded,
                               'persons': len(face_engine.known_encodings)
                           })

@app.route('/user/dashboard')
def user_dashboard():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('user_dashboard.html', user=session)

# ── Routes: Auth ───────────────────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.json
    required = ['username', 'email', 'password', 'full_name']
    for r in required:
        if not d.get(r):
            return jsonify({'error': f'{r} is required'}), 400

    with get_db() as db:
        existing = db.execute(
            'SELECT id FROM users WHERE username=? OR email=?',
            (d['username'], d['email'])).fetchone()
        if existing:
            return jsonify({'error': 'Username or email already exists'}), 400

        uid = str(uuid.uuid4())
        colors = ['#6C63FF','#FF6584','#43E97B','#F9A825','#00BCD4','#FF8C42']
        import random
        db.execute("""INSERT INTO users
            (id,username,email,password_hash,role,approved,created_at,
             full_name,phone,department,avatar_color)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (uid, d['username'], d['email'], hash_pw(d['password']),
             'user', 0, now(), d['full_name'],
             d.get('phone',''), d.get('department',''),
             random.choice(colors)))
        db.commit()

    logging.info(f"New registration: {d['username']} - pending admin approval")
    return jsonify({'success': True,
                    'message': 'Registration successful! Awaiting admin approval.'})

@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.json
    with get_db() as db:
        user = db.execute(
            'SELECT * FROM users WHERE username=? OR email=?',
            (d.get('username',''), d.get('username',''))).fetchone()
        if not user or user['password_hash'] != hash_pw(d.get('password','')):
            return jsonify({'error': 'Invalid credentials'}), 401
        if not user['approved']:
            return jsonify({'error': 'Account pending admin approval. Please wait.'}), 403

        db.execute('UPDATE users SET last_login=? WHERE id=?',
                   (now(), user['id']))
        db.commit()

    session.clear()
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    session['full_name'] = user['full_name']
    session['avatar_color'] = user['avatar_color'] or '#6C63FF'

    return jsonify({
        'success': True,
        'role': user['role'],
        'redirect': '/admin/dashboard' if user['role'] == 'admin' else '/user/dashboard'
    })

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

# ── Routes: Admin - User Management ───────────────────────────
@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_users():
    status = request.args.get('status', 'all')
    with get_db() as db:
        if status == 'pending':
            users = db.execute(
                'SELECT * FROM users WHERE approved=0 ORDER BY created_at DESC').fetchall()
        elif status == 'approved':
            users = db.execute(
                'SELECT * FROM users WHERE approved=1 ORDER BY created_at DESC').fetchall()
        else:
            users = db.execute(
                'SELECT * FROM users ORDER BY created_at DESC').fetchall()
    return jsonify([dict(u) for u in users])

@app.route('/api/admin/users/<user_id>/approve', methods=['POST'])
@admin_required
def approve_user(user_id):
    with get_db() as db:
        db.execute(
            'UPDATE users SET approved=1, approved_at=?, approved_by=? WHERE id=?',
            (now(), session['username'], user_id))
        db.commit()
        user = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    logging.info(f"User {user['username']} approved by {session['username']}")
    return jsonify({'success': True, 'message': f"User {user['username']} approved"})

@app.route('/api/admin/users/<user_id>/reject', methods=['POST'])
@admin_required
def reject_user(user_id):
    with get_db() as db:
        db.execute('DELETE FROM users WHERE id=? AND role!=?', (user_id, 'admin'))
        db.commit()
    return jsonify({'success': True})

@app.route('/api/admin/users/<user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    with get_db() as db:
        user = db.execute('SELECT approved FROM users WHERE id=?', (user_id,)).fetchone()
        new_status = 0 if user['approved'] else 1
        db.execute('UPDATE users SET approved=? WHERE id=?', (new_status, user_id))
        db.commit()
    return jsonify({'success': True, 'approved': new_status})

# ── Routes: Face Registration ──────────────────────────────────
@app.route('/api/faces/register', methods=['POST'])
@admin_required
def register_face():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    person_name = request.form.get('person_name', '').strip()
    person_id = request.form.get('person_id', str(uuid.uuid4())[:8])
    category = request.form.get('category', 'staff')
    is_authorized = request.form.get('is_authorized', 'true') == 'true'

    if not person_name:
        return jsonify({'error': 'Person name required'}), 400

    files = request.files.getlist('images')
    all_encodings = []
    saved_paths = []
    errors = []

    person_dir = FACE_DATA_DIR / person_id
    person_dir.mkdir(exist_ok=True)

    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            fpath = person_dir / fname
            f.save(str(fpath))
            saved_paths.append(str(fpath))

            if FR_AVAILABLE:
                encs = face_engine.encode_image(fpath)
                if encs:
                    all_encodings.extend([e.tolist() for e in encs])
                else:
                    errors.append(f"No face detected in {fname}")
        else:
            errors.append(f"Invalid file: {f.filename}")

    if all_encodings or not FR_AVAILABLE:
        enc_arr = np.array(all_encodings) if all_encodings else np.array([])
        np.save(str(FACE_DATA_DIR / f"{person_id}.npy"), enc_arr)
        meta = {
            'person_id': person_id,
            'person_name': person_name,
            'is_authorized': is_authorized,
            'category': category,
            'image_count': len(files),
            'encoding_count': len(all_encodings),
            'registered_at': now(),
            'registered_by': session['username']
        }
        (FACE_DATA_DIR / f"{person_id}_meta.json").write_text(json.dumps(meta))

        with get_db() as db:
            db.execute("""INSERT OR REPLACE INTO registered_faces
                (id,person_id,person_name,image_path,encoding_path,
                 registered_by,registered_at,is_authorized,category)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), person_id, person_name,
                 str(person_dir), str(FACE_DATA_DIR / f"{person_id}.npy"),
                 session['username'], now(), int(is_authorized), category))
            db.commit()

        face_engine.load_encodings()
        return jsonify({
            'success': True,
            'person_id': person_id,
            'images_processed': len(files),
            'encodings_extracted': len(all_encodings),
            'errors': errors
        })

    return jsonify({'error': 'No valid face encodings extracted', 'details': errors}), 400

@app.route('/api/faces/list', methods=['GET'])
@login_required
def list_faces():
    with get_db() as db:
        faces = db.execute(
            'SELECT * FROM registered_faces ORDER BY registered_at DESC').fetchall()
    return jsonify([dict(f) for f in faces])

@app.route('/api/faces/<person_id>/toggle', methods=['POST'])
@admin_required
def toggle_face_auth(person_id):
    meta_file = FACE_DATA_DIR / f"{person_id}_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        meta['is_authorized'] = not meta.get('is_authorized', True)
        meta_file.write_text(json.dumps(meta))
    with get_db() as db:
        db.execute(
            'UPDATE registered_faces SET is_authorized=? WHERE person_id=?',
            (int(not meta.get('is_authorized', True)), person_id))
        db.commit()
    face_engine.load_encodings()
    return jsonify({'success': True})

@app.route('/api/faces/<person_id>', methods=['DELETE'])
@admin_required
def delete_face(person_id):
    for ext in ['.npy', '_meta.json']:
        f = FACE_DATA_DIR / f"{person_id}{ext}"
        if f.exists(): f.unlink()
    with get_db() as db:
        db.execute('DELETE FROM registered_faces WHERE person_id=?', (person_id,))
        db.commit()
    face_engine.load_encodings()
    return jsonify({'success': True})

# ── Routes: Training ───────────────────────────────────────────
@app.route('/api/train/upload', methods=['POST'])
@admin_required
def upload_training_data():
    """Upload dataset zip or folder structure."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files'}), 400

    files = request.files.getlist('files')
    saved = 0
    persons = {}

    for f in files:
        if not allowed_file(f.filename): continue
        # Path: person_name/image.jpg
        parts = f.filename.replace('\\','/').split('/')
        if len(parts) >= 2:
            person_name = parts[-2]
            img_name = secure_filename(parts[-1])
        else:
            person_name = 'unknown'
            img_name = secure_filename(f.filename)

        person_id = hashlib.md5(person_name.encode()).hexdigest()[:12]
        person_dir = FACE_DATA_DIR / person_id
        person_dir.mkdir(exist_ok=True)

        save_path = person_dir / img_name
        f.save(str(save_path))
        saved += 1
        persons[person_id] = person_name

    return jsonify({
        'success': True,
        'files_saved': saved,
        'persons_found': len(persons),
        'persons': persons
    })

@app.route('/api/train/start', methods=['POST'])
@admin_required
def start_training():
    global training_state
    if training_state['running']:
        return jsonify({'error': 'Training already in progress'}), 400

    d = request.json or {}
    model_type = d.get('model_type', 'face_recognition')

    def run_training():
        global training_state
        training_state = {
            'running': True, 'progress': 0, 'total': 0,
            'current': 'Scanning face database...', 'done': False, 'result': None
        }
        session_id = str(uuid.uuid4())

        with get_db() as db:
            db.execute("""INSERT INTO training_sessions
                (id,started_at,status,model_type)
                VALUES (?,?,?,?)""",
                (session_id, now(), 'running', model_type))
            db.commit()

        try:
            # Load all face data from face_data directory
            person_dirs = [d for d in FACE_DATA_DIR.iterdir()
                          if d.is_dir()]
            training_state['total'] = len(person_dirs)

            total_encodings = 0
            total_persons = 0

            for idx, person_dir in enumerate(person_dirs):
                person_id = person_dir.name
                meta_file = FACE_DATA_DIR / f"{person_id}_meta.json"

                if not meta_file.exists():
                    # Auto-detect person name from directory
                    person_name = person_dir.name
                    meta = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'is_authorized': True,
                        'category': 'trained',
                    }
                else:
                    meta = json.loads(meta_file.read_text())

                training_state['current'] = f"Training: {meta.get('person_name', person_id)}"
                training_state['progress'] = idx + 1

                img_files = (list(person_dir.glob('*.jpg')) +
                            list(person_dir.glob('*.jpeg')) +
                            list(person_dir.glob('*.png')))

                all_encs = []
                for img_path in img_files:
                    try:
                        encs = face_engine.encode_image(img_path)
                        if encs:
                            all_encs.extend([e.tolist() for e in encs])
                    except Exception as e:
                        logging.error(f"Training error on {img_path}: {e}")

                if all_encs:
                    np.save(str(FACE_DATA_DIR / f"{person_id}.npy"),
                            np.array(all_encs))
                    meta['encoding_count'] = len(all_encs)
                    meta['trained_at'] = now()
                    meta_file.write_text(json.dumps(meta))
                    total_encodings += len(all_encs)
                    total_persons += 1

            face_engine.load_encodings()

            result = {
                'total_persons': total_persons,
                'total_encodings': total_encodings,
                'status': 'completed'
            }
            training_state.update({
                'running': False, 'done': True,
                'current': 'Training Complete!', 'result': result
            })

            with get_db() as db:
                db.execute("""UPDATE training_sessions
                    SET completed_at=?,status=?,total_images=?,total_persons=?
                    WHERE id=?""",
                    (now(), 'completed', total_encodings, total_persons, session_id))
                db.commit()

        except Exception as e:
            training_state.update({
                'running': False, 'done': True,
                'current': f'Error: {str(e)}', 'result': {'error': str(e)}
            })
            logging.error(f"Training error: {e}")

    t = threading.Thread(target=run_training)
    t.daemon = True
    t.start()
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/api/train/status', methods=['GET'])
@login_required
def train_status():
    return jsonify(training_state)

# ── Routes: Recognition ────────────────────────────────────────
@app.route('/api/recognize/frame', methods=['POST'])
@login_required
def recognize_frame():
    """Recognize face from base64 frame (live camera)."""
    d = request.json
    image_b64 = d.get('image', '')
    gate = d.get('gate', 'Main Gate')

    if not image_b64:
        return jsonify({'error': 'No image'}), 400

    try:
        img_bytes = base64.b64decode(image_b64.split(',')[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)

        if CV2_AVAILABLE:
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            frame_rgb = np.array(pil_img)

        results = face_engine.recognize_frame(frame_rgb)

        for r in results:
            log_entry(r, gate, 'live_camera')

        return jsonify({'success': True, 'results': results, 'count': len(results)})
    except Exception as e:
        logging.error(f"Frame recognition error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize/upload', methods=['POST'])
@login_required
def recognize_upload():
    """Recognize faces from multiple uploaded images."""
    if 'images' not in request.files:
        return jsonify({'error': 'No images'}), 400

    gate = request.form.get('gate', 'Upload Gate')
    files = request.files.getlist('images')
    all_results = []

    for f in files:
        if not f or not allowed_file(f.filename):
            continue
        try:
            fname = secure_filename(f.filename)
            save_path = UPLOAD_DIR / f"{uuid.uuid4()}_{fname}"
            f.save(str(save_path))

            img_bytes = save_path.read_bytes()
            nparr = np.frombuffer(img_bytes, np.uint8)

            if CV2_AVAILABLE:
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                pil_img = Image.open(str(save_path)).convert('RGB')
                frame_rgb = np.array(pil_img)

            results = face_engine.recognize_frame(frame_rgb)

            # Annotate image
            img_b64 = ''
            if CV2_AVAILABLE and results:
                for r in results:
                    bb = r['bbox']
                    color = (0, 255, 100) if r['authorized'] else (0, 60, 255)
                    cv2.rectangle(frame, (bb['left'], bb['top']),
                                  (bb['right'], bb['bottom']), color, 2)
                    label = f"{r['person_name']} {r['confidence']:.1f}%"
                    cv2.putText(frame, label,
                                (bb['left'], bb['top'] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_b64 = base64.b64encode(buf.tobytes()).decode()

            for r in results:
                log_entry(r, gate, 'image_upload')

            all_results.append({
                'filename': fname,
                'results': results,
                'annotated_image': img_b64,
                'face_count': len(results)
            })

            # Clean up
            save_path.unlink(missing_ok=True)

        except Exception as e:
            all_results.append({'filename': f.filename, 'error': str(e), 'results': []})

    return jsonify({
        'success': True,
        'files_processed': len(all_results),
        'results': all_results
    })

def log_entry(result, gate, method):
    """Save recognition result to database."""
    log_id = str(uuid.uuid4())
    status = 'authorized' if result['authorized'] else 'unauthorized'

    with get_db() as db:
        db.execute("""INSERT INTO entry_logs
            (id,person_id,person_name,status,confidence,gate,timestamp,detection_method)
            VALUES (?,?,?,?,?,?,?,?)""",
            (log_id, result['person_id'], result['person_name'],
             status, result['confidence'], gate, now(), method))
        db.commit()

    if not result['authorized']:
        alert_id = str(uuid.uuid4())
        with get_db() as db:
            db.execute("""INSERT INTO alerts
                (id,type,message,person_name,gate,timestamp)
                VALUES (?,?,?,?,?,?)""",
                (alert_id, 'unauthorized',
                 f"Unauthorized access attempt at {gate}",
                 result['person_name'], gate, now()))
            db.commit()

# ── Routes: Logs & Stats ───────────────────────────────────────
@app.route('/api/logs', methods=['GET'])
@login_required
def get_logs():
    status = request.args.get('status', '')
    gate = request.args.get('gate', '')
    limit = int(request.args.get('limit', 100))

    query = 'SELECT * FROM entry_logs WHERE 1=1'
    params = []
    if status:
        query += ' AND status=?'; params.append(status)
    if gate:
        query += ' AND gate=?'; params.append(gate)
    query += ' ORDER BY timestamp DESC LIMIT ?'
    params.append(limit)

    with get_db() as db:
        logs = db.execute(query, params).fetchall()
    return jsonify([dict(l) for l in logs])

@app.route('/api/logs/<log_id>/override', methods=['POST'])
@admin_required
def override_log(log_id):
    d = request.json
    new_status = 'authorized' if d.get('allow') else 'unauthorized'
    with get_db() as db:
        db.execute("""UPDATE entry_logs
            SET status=?,admin_override=1,override_by=?,notes=?
            WHERE id=?""",
            (new_status, session['username'], d.get('note',''), log_id))
        db.commit()
    return jsonify({'success': True})

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    with get_db() as db:
        total = db.execute('SELECT COUNT(*) as c FROM entry_logs').fetchone()['c']
        auth = db.execute("SELECT COUNT(*) as c FROM entry_logs WHERE status='authorized'").fetchone()['c']
        unauth = db.execute("SELECT COUNT(*) as c FROM entry_logs WHERE status='unauthorized'").fetchone()['c']
        persons = db.execute('SELECT COUNT(*) as c FROM registered_faces').fetchone()['c']
        pending = db.execute("SELECT COUNT(*) as c FROM users WHERE approved=0").fetchone()['c']
        alerts = db.execute("SELECT COUNT(*) as c FROM alerts WHERE resolved=0").fetchone()['c']
        recent = db.execute(
            'SELECT * FROM entry_logs ORDER BY timestamp DESC LIMIT 10').fetchall()
        hourly = db.execute("""
            SELECT strftime('%H', timestamp) as hr, COUNT(*) as cnt
            FROM entry_logs
            WHERE date(timestamp) = date('now')
            GROUP BY hr ORDER BY hr""").fetchall()

    return jsonify({
        'total_entries': total,
        'authorized': auth,
        'unauthorized': unauth,
        'registered_persons': persons,
        'pending_users': pending,
        'active_alerts': alerts,
        'engine_loaded': face_engine.is_loaded,
        'known_persons': len(face_engine.known_encodings),
        'recent_entries': [dict(r) for r in recent],
        'hourly': [dict(h) for h in hourly]
    })

@app.route('/api/alerts', methods=['GET'])
@login_required
def get_alerts():
    with get_db() as db:
        alerts = db.execute(
            'SELECT * FROM alerts WHERE resolved=0 ORDER BY timestamp DESC LIMIT 50').fetchall()
    return jsonify([dict(a) for a in alerts])

@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
@admin_required
def resolve_alert(alert_id):
    with get_db() as db:
        db.execute('UPDATE alerts SET resolved=1,resolved_by=? WHERE id=?',
                   (session['username'], alert_id))
        db.commit()
    return jsonify({'success': True})

@app.route('/api/system/load-engine', methods=['POST'])
@admin_required
def load_engine():
    face_engine.load_encodings()
    return jsonify({
        'success': True,
        'persons_loaded': len(face_engine.known_encodings)
    })

@app.route('/api/training/history', methods=['GET'])
@login_required
def training_history():
    with get_db() as db:
        sessions = db.execute(
            'SELECT * FROM training_sessions ORDER BY started_at DESC LIMIT 20').fetchall()
    return jsonify([dict(s) for s in sessions])

if __name__ == '__main__':
    init_db()
    face_engine.load_encodings()
    print("=" * 55)
    print("  SENTINEL PRO - Advanced AI Security System")
    print("=" * 55)
    print(f"  OpenCV       : {'✓' if CV2_AVAILABLE else '✗ (install: pip install opencv-python)'}")
    print(f"  face_recog   : {'✓' if FR_AVAILABLE else '✗ (install: pip install face_recognition)'}")
    print(f"  DeepFace     : {'✓' if DF_AVAILABLE else '✗ (optional)'}")
    print(f"  Known persons: {len(face_engine.known_encodings)}")
    print(f"  Server       : http://localhost:5000")
    print("=" * 55)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
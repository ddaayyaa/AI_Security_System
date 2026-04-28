# 🛡️ SENTINEL PRO — Advanced AI Security System

Full-stack AI-powered event security system with facial recognition, dual login, admin approval workflow, live camera, bulk image recognition, and model training.

---

## 🚀 Quick Start

### Windows
```cmd
Double-click start.bat
```
Then open: **http://localhost:5000**

### Manual
```cmd
pip install flask Pillow numpy opencv-python
pip install face-recognition    ← needs CMake first
python app.py
```

---

## 🔐 Login Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| User | Register first, then admin approves |

---

## 🗂️ Project Structure

```
sentinel_pro/
├── app.py                    ← Full Flask backend
├── start.bat                 ← One-click Windows launch
├── requirements.txt
├── templates/
│   ├── index.html            ← Login / Register page
│   ├── admin_dashboard.html  ← Full admin control center
│   └── user_dashboard.html   ← Normal user portal
├── database/
│   └── sentinel.db           ← SQLite database (auto-created)
├── face_data/                ← Face encodings (.npy) + metadata
├── uploads/                  ← Temporary upload storage
├── models/                   ← Saved ML models
└── logs/                     ← Server logs
```

---

## ✨ Features

### 🔐 Dual Authentication
- **Admin login** — Full access to all features
- **User login** — Camera + image upload only
- **Registration** — New users register and await admin approval
- Admin can approve, disable, or delete user accounts

### 📹 Live Camera Recognition
- Real-time face detection every 2.5 seconds
- Shows name, confidence score, authorization status
- Session statistics (total / authorized / unauthorized)
- Works from any device with a camera (mobile friendly)

### 🖼️ Bulk Image Recognition
- Upload unlimited images at once
- Drag and drop supported
- Annotated result images with bounding boxes
- Summary stats + per-image detailed reports

### 🎯 AI Model Training
- Upload dataset organized as `person_name/image.jpg`
- Supports large datasets (thousands of images)
- Real-time training progress bar
- Training history log

### 👤 Face Registration
- Register individual persons with multiple photos
- Set authorized/unauthorized status
- Categories: Staff, VIP, Vendor, Security, Guest
- Instantly updates the recognition engine

### 👥 User Management (Admin)
- View all pending / approved users
- Approve or reject registrations with one click
- Enable/disable access without deleting account

### 📋 Entry Logs
- Full history of all recognition events
- Filter by status (authorized / unauthorized)
- Admin override for wrong decisions
- Method tracking (live camera vs image upload)

### 🚨 Alert System
- Automatic alerts for unauthorized access
- Alert management dashboard
- Resolve alerts with one click

---

## 🧠 Face Recognition Engine

### Mode 1: Demo Mode (no installation needed)
- Simulates recognition results
- All other features work fully
- Good for testing the UI

### Mode 2: face_recognition (Recommended)
```cmd
:: Windows — Install CMake first from cmake.org
pip install cmake
pip install dlib
pip install face-recognition
```

### Mode 3: DeepFace (Deep Learning)
```cmd
pip install deepface tensorflow
```

---

## 📂 Dataset Format for Training

```
dataset/
├── John_Smith/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── Jane_Doe/
│   ├── img1.jpg
│   └── img2.jpg
└── ...
```

**Tips for best accuracy:**
- 5-15 images per person
- Different angles and lighting
- Clear, unobstructed face
- Mix of indoor and outdoor photos

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/login` | Login |
| POST | `/api/auth/register` | Register account |
| POST | `/api/auth/logout` | Logout |
| GET | `/api/admin/users` | List users |
| POST | `/api/admin/users/:id/approve` | Approve user |
| POST | `/api/faces/register` | Register face |
| GET | `/api/faces/list` | List faces |
| POST | `/api/recognize/frame` | Recognize from camera |
| POST | `/api/recognize/upload` | Recognize uploaded images |
| POST | `/api/train/upload` | Upload training data |
| POST | `/api/train/start` | Start training |
| GET | `/api/train/status` | Training progress |
| GET | `/api/logs` | Entry logs |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/alerts` | Active alerts |

---

## 🔧 Configuration

Edit top of `app.py`:
```python
app.secret_key = 'your-secret-key'  # Change in production
```

Change admin password:
```python
# Default: admin123 — change via SQLite browser or add a route
```

---

## 📱 Mobile Access

The system works on mobile browsers:
1. Find your PC's IP: `ipconfig` → look for IPv4
2. Open on phone: `http://192.168.x.x:5000`
3. Camera works on mobile too (HTTPS needed for production)

---

## 🔒 Production Deployment

For events/production:
1. Use HTTPS (nginx + SSL certificate)
2. Change secret key
3. Use PostgreSQL instead of SQLite
4. Add rate limiting
5. Set up proper CORS

---

© 2024 SENTINEL PRO — Advanced AI Security System

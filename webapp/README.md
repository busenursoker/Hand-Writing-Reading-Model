# Handwriting OCR Web App (Flutter + FastAPI)

This project is a Flutter **web** frontend for a CNN-based handwriting recognition system.
The frontend communicates with a Python **FastAPI** backend that loads the trained CNN model and returns predictions.

## Tech Stack

* **Frontend:** Flutter (Web)
* **Backend:** Python, FastAPI, Uvicorn
* **Model:** CNN trained on EMNIST (TensorFlow/Keras)

---

## Requirements

### Frontend

* Flutter SDK installed
* Web enabled for Flutter

### Backend

* Python 3.9+ recommended
* FastAPI + Uvicorn installed (plus your ML dependencies)

---

## Run the Backend (FastAPI)

From the backend directory (where `api.py` exists):

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at:

* API: `http://localhost:8000`
* Interactive docs (if enabled): `http://localhost:8000/docs`

---

## Run the Frontend (Flutter Web)

### 1) Add Flutter to PATH (if needed)

```bash
export PATH="$HOME/flutter/bin:$PATH"
```

### 2) Enable Flutter web support (one-time)

```bash
flutter config --enable-web
```

### 3) Install dependencies

From the Flutter project root:

```bash
flutter pub get
```

### 4) Run on web-server

```bash
flutter run -d web-server --web-port 8080
```

Frontend will be available at:

* `http://localhost:8080`

---

## Notes / Common Issues

### CORS (Frontend ↔ Backend)

If your Flutter web app calls `http://localhost:8000`, your backend must allow CORS.

In FastAPI, you typically need something like:

* allow origin: `http://localhost:8080`
* allow methods/headers: `*`

(Exact setup depends on how your backend is written.)

### Ports

* Flutter web: **8080**
* FastAPI: **8000**

If you change ports, update the frontend API base URL accordingly.

---

## Useful Commands

### Flutter

```bash
flutter clean
flutter pub get
flutter run -d web-server --web-port 8080
```

### Backend

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

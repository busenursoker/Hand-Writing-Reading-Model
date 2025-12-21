# 📝 Hand Writing Reading Model

A simple and effective application that converts handwritten text into digital text using **OCR (Optical Character Recognition)**. This project consists of a **FastAPI** backend and a **Flutter** mobile application.

## 🌟 Features

* **Handwriting Recognition:** Capture or upload images of handwritten text.
* **FastAPI Backend:** Fast and scalable Python API for processing images.
* **Flutter Frontend:** Cross-platform mobile UI for a smooth user experience.
* **Tesseract OCR:** Uses Google's Tesseract engine for high-accuracy text extraction.

---

## 🛠️ Tech Stack

* **Frontend:** Flutter (Dart)
* **Backend:** FastAPI (Python)
* **OCR Engine:** Tesseract OCR
* **Image Processing:** OpenCV / Pillow

---

## 🚀 How to Run

### 1. Backend (FastAPI)

Navigate to the backend folder, install dependencies, and start the server:

```bash
# Install requirements
pip install fastapi uvicorn pytesseract opencv-python

# Run the server
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

```

> **Note:** Make sure **Tesseract OCR** is installed on your system.

### 2. Frontend (Flutter)

Navigate to the flutter folder and run the app:

```bash

# Run the app
flutter run -d web-server --web-port 8080

```


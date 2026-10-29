# Face Detection & Recognition System

A real-time **Face Detection and Recognition System** built in Python using OpenCV (`opencv-contrib-python`) and NumPy. Developed as part of the Artificial Intelligence internship at **Syntecxhub**.

---

## Features

- **Register / Add a New Person (Webcam):** Captures multiple facial samples via webcam, automatically detects faces using Haar Cascades, and trains/updates the recognition model.
- **Live Face Detection & Recognition (Webcam):** Performs real-time face detection and identifies registered individuals with bounding boxes, names, and confidence scores.
- **Test Face Recognition on an Image File:** Evaluates static image files to detect and recognize faces.
- **View Registered People:** Displays all registered users stored in the local dataset.
- **Automated Model Training:** Uses the LBPH (Local Binary Patterns Histograms) Face Recognizer (`cv2.face.LBPHFaceRecognizer_create()`) for fast, local training and inference.

---

## Tech Stack

- **Language:** Python 3.13
- **Libraries:**
  - `opencv-contrib-python` (v4.10.0.84) — Haar Cascade Classifier & LBPH Face Recognizer
  - `numpy` (>=2.0.0) — Matrix & image array processing

---

## Project Structure

```text
Syntecxhub_Face_Recognition/
├── face_recognition_system.py   # Main application script
├── requirements.txt             # Required Python dependencies
└── README.md                    # Project documentation

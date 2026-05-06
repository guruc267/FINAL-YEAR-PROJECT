# 🍎 FreshAgent — AI-Based Fruits & Vegetables Adulteration Detection System

> **A deep learning-powered food safety system** that detects whether fruits and vegetables are **Fresh**, **Rotten**, or **Formalin-treated** using EfficientNetV2 and real-time inference via a web dashboard.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Prerequisites](#-prerequisites)
6. [Setup & Installation](#-setup--installation)
7. [Running the Application](#-running-the-application)
8. [Training the Model (Optional)](#-training-the-model-optional)
9. [Evaluating the Model](#-evaluating-the-model)
10. [ESP32-CAM Integration](#-esp32-cam-integration)
11. [API Reference](#-api-reference)
12. [Supported Classes](#-supported-classes)
13. [Dataset Structure](#-dataset-structure)
14. [Screenshots](#-screenshots)
15. [Tech Stack](#-tech-stack)
16. [Team & Acknowledgements](#-team--acknowledgements)
17. [Disclaimer](#-disclaimer)

---

## 🎯 Project Overview

**FreshAgent** is an AI-powered food adulteration detection system that uses deep learning to classify fruits and vegetables as Fresh, Rotten, or Formalin-Mixed (adulterated). The system uses:

- **EfficientNetV2-B0** as the backbone CNN with a multi-task classification head (predicts both the produce type and its condition simultaneously).
- **ONNX Runtime** for fast, optimized CPU inference on the server.
- **FastAPI** as the backend web framework serving a real-time web dashboard.
- **Grad-CAM** for Explainable AI (XAI) — visually highlights which regions of the image the model uses for its prediction.
- **ESP32-CAM / Phone Camera** integration for live hardware-based detection.

### Problem Statement

Food adulteration, particularly the use of harmful chemicals like **formalin** to artificially preserve fruits, is a growing public health concern in India. FreshAgent addresses this by providing an automated, real-time visual detection system that can flag potentially adulterated produce.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Task Classification** | Predicts produce type (Apple, Banana, Mango, Orange, Grape) AND condition (Fresh, Rotten, Formalin-Mixed) in a single forward pass |
| 📊 **Grad-CAM Heatmaps** | Visual explanation of model predictions — shows which image regions influenced the decision |
| 🌐 **Web Dashboard** | Upload images via browser and get instant results with confidence scores |
| 📷 **ESP32-CAM Support** | Connect an ESP32-CAM module for automated, continuous monitoring |
| 📱 **Phone Camera Mode** | Use your phone as a camera via QR code — no app installation needed |
| 🥕 **Vegetable Module** | Separate VeggieAgent model for vegetable adulteration detection (Ginger) |
| ⚡ **ONNX Optimized** | Inference runs on ONNX Runtime for maximum CPU speed |

---

## 🏗 System Architecture

```
┌─────────────────┐     JPEG      ┌──────────────────────────────┐
│   ESP32-CAM     │──────────────►│                              │
│  (IoT Device)   │   HTTP POST   │     FastAPI Backend          │
└─────────────────┘               │     (Python + ONNX)          │
                                  │                              │
┌─────────────────┐     JPEG      │  ┌────────────────────────┐  │
│   Phone Camera  │──────────────►│  │  ONNX Inference Engine │  │
│  (via Browser)  │   HTTP POST   │  │  (EfficientNetV2-B0)   │  │
└─────────────────┘               │  └────────────────────────┘  │
                                  │                              │
┌─────────────────┐    Upload     │  ┌────────────────────────┐  │
│   Web Browser   │──────────────►│  │  Grad-CAM (PyTorch)    │  │
│   (Dashboard)   │◄──────────────│  │  Explainability Layer  │  │
└─────────────────┘   JSON+CAM    │  └────────────────────────┘  │
                                  └──────────────────────────────┘
```

**Key Design Decisions:**
- All AI inference runs **server-side** — the ESP32-CAM only captures and sends images.
- The ONNX model is used for fast inference; the PyTorch `.pth` checkpoint is loaded separately **only** for Grad-CAM heatmap generation.
- Multi-task learning allows a single model to predict both the produce type and adulteration condition.

---

## 📁 Project Structure

```
FreshAgent-Submission/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── start_server.bat             ← One-click server launch (Windows)
├── train.bat                    ← One-click model training (Fruits)
├── train_veggie.bat             ← One-click model training (Vegetables)
├── train_overnight.bat          ← Extended overnight training script
├── auto_stop.py                 ← Auto-stop training after N epochs
├── evaluate.py                  ← Full model evaluation with 7 graph types
│
├── backend/                     ← FastAPI server & inference engine
│   ├── main.py                  ← FastAPI app — routes, endpoints, startup
│   ├── inference.py             ← ONNX inference + Grad-CAM (Fruits)
│   └── veggie_inference.py      ← ONNX inference + Grad-CAM (Vegetables)
│
├── frontend/                    ← Web dashboard (HTML/CSS/JS)
│   ├── index.html               ← Main dashboard page
│   ├── style.css                ← Dashboard styling
│   ├── app.js                   ← Dashboard logic (upload, polling, rendering)
│   └── phone_cam.html           ← Phone camera capture page
│
├── training/                    ← Model training code (Fruits)
│   ├── dataset.py               ← Dataset loader, augmentations, stratified split
│   ├── model.py                 ← EfficientNetV2-B0 multi-task architecture
│   ├── train.py                 ← Training loop (2-phase, MixUp, Early Stopping)
│   ├── export.py                ← PyTorch → ONNX export
│   └── plot_results.py          ← Training curve & confusion matrix plots
│
├── veggie_training/             ← Model training code (Vegetables)
│   ├── veggie_dataset.py        ← Vegetable dataset loader
│   ├── veggie_model.py          ← Vegetable model architecture
│   ├── train.py                 ← Vegetable training loop
│   └── export.py                ← Vegetable model ONNX export
│
├── esp32cam/                    ← ESP32-CAM Arduino firmware
│   └── FreshAgent_ESP32CAM.ino  ← Complete ESP32-CAM C++ code
│
└── models/                      ← Pre-trained model files
    ├── freshagent.onnx          ← Fruit model (ONNX, ~25 MB)
    ├── veggieagent.onnx         ← Vegetable model (ONNX, ~25 MB)
    ├── best_model.pth           ← Fruit model (PyTorch, for Grad-CAM)
    ├── veggieagent_best.pth     ← Veggie model (PyTorch, for Grad-CAM)
    └── eval_plots/              ← Evaluation graphs (confusion matrices, ROC, etc.)
```

---

## 📌 Prerequisites

| Requirement | Version |
|-------------|---------|
| **Python** | 3.10 or higher |
| **pip** | Latest |
| **OS** | Windows 10/11 (tested), Linux/macOS (should work with minor path changes) |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Storage** | ~500 MB for code + models (5 GB+ if training with dataset) |
| **GPU** | Optional — Intel Iris Xe via DirectML, or NVIDIA CUDA. CPU works fine for inference. |

### Verify Python Installation
```powershell
python --version
# Should print: Python 3.10.x or higher
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/) and ensure **"Add Python to PATH"** is checked during installation.

---

## 🚀 Setup & Installation

### Method 1: One-Click Setup (Recommended for Windows)

Simply double-click **`start_server.bat`** — it will:
1. ✅ Create a Python virtual environment (`venv/`)
2. ✅ Install all dependencies from `requirements.txt`
3. ✅ Start the FastAPI server

### Method 2: Manual Setup

Open a terminal (PowerShell or Command Prompt) in the project directory:

```powershell
# Step 1: Create a virtual environment
python -m venv venv

# Step 2: Activate it
venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt
```

> **Note for Linux/macOS users:**
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

### What Gets Installed?

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `timm` | EfficientNetV2 pretrained weights |
| `onnxruntime` | Fast ONNX model inference |
| `fastapi`, `uvicorn` | Web server backend |
| `Pillow`, `opencv-python` | Image processing |
| `scikit-learn` | Evaluation metrics |
| `pytorch-grad-cam` | Explainable AI heatmaps |
| `albumentations` | Data augmentation (training) |
| `matplotlib`, `seaborn` | Plot generation |
| `qrcode` | QR code generation for phone camera |

---

## ▶️ Running the Application

### Option A: Using Batch Script (Windows)

```
start_server.bat
```

### Option B: Manual Command

```powershell
# Activate the virtual environment first
venv\Scripts\activate

# Start the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 9090 --reload
```

### Access the Application

Once the server is running, open your browser and navigate to:

| URL | Description |
|-----|-------------|
| **http://localhost:9090** | 🖥️ Main Web Dashboard |
| **http://localhost:9090/docs** | 📖 Interactive API Documentation (Swagger UI) |
| **http://localhost:9090/phone-cam** | 📱 Phone Camera Page |

### Using the Web Dashboard

1. Open **http://localhost:9090** in your browser.
2. Click **"Upload Image"** and select a fruit/vegetable image.
3. The system will display:
   - **Predicted produce type** (e.g., Apple, Banana, Mango)
   - **Condition** (Fresh / Rotten / Formalin-Mixed)
   - **Confidence scores** for each class
   - **Grad-CAM heatmap** showing which regions the model focused on
   - **Safety verdict** (SAFE / UNSAFE / DANGER)

### Using Phone Camera

1. Open the dashboard on your PC.
2. Scan the **QR code** displayed on the dashboard with your phone.
3. Your phone browser will open the camera capture page.
4. Point the phone camera at fruit/vegetables — results appear live on the PC dashboard.

> **Note:** Both devices must be on the **same WiFi network**.

---

## 🧠 Training the Model (Optional)

Pre-trained models are included in the `models/` folder. You only need to retrain if you want to improve accuracy or add new classes.

### Dataset Required

You need a structured dataset with this folder layout:

```
YourDataset/
├── Apple/
│   ├── Fresh/          ← images (.jpg, .png)
│   ├── Rotten/
│   └── Formalin_Mixed/
├── Banana/
│   ├── Fresh/
│   ├── Rotten/
│   └── Formalin_Mixed/
├── Grape/
│   └── ...
├── Mango/
│   └── ...
└── Orange/
    └── ...
```

### Train Using Batch Script

```
train.bat "C:\path\to\your\dataset"
```

This will:
1. Train EfficientNetV2-B0 for up to 60 epochs (with early stopping)
2. Export the best model to ONNX automatically
3. Generate training curves and confusion matrices

### Train Manually

```powershell
venv\Scripts\activate

# Train
python training\train.py --data_dir "C:\path\to\dataset" --epochs 60 --batch_size 16

# Export to ONNX
python training\export.py --checkpoint models\best_model.pth --output models\freshagent.onnx

# Generate plots
python training\plot_results.py --output_dir models
```

### Training Time Estimates (Intel Iris Xe, batch_size=16)

| Epochs | Estimated Time |
|--------|---------------|
| 10     | ~15–25 min    |
| 30     | ~45–75 min    |
| 60     | ~90–150 min   |

> **Tip:** If you run out of memory, reduce batch size: `--batch_size 8`

---

## 📊 Evaluating the Model

Run the full evaluation script to generate comprehensive performance graphs:

```powershell
venv\Scripts\activate
python evaluate.py --data_dir "path\to\dataset" --model_path models\best_model.pth --output_dir models\eval_plots
```

This generates **7 evaluation graphs**:

| # | Graph | Description |
|---|-------|-------------|
| 1 | Confusion Matrix (Condition) | Fresh/Rotten/Formalin classification accuracy |
| 2 | Confusion Matrix (Fruit) | Fruit type identification accuracy |
| 3 | Per-Class Metrics (Condition) | Precision, Recall, F1 for each condition |
| 4 | Per-Class Metrics (Fruit) | Precision, Recall, F1 for each fruit |
| 5 | ROC Curves | One-vs-Rest ROC curves with AUC scores |
| 6 | Confidence Distribution | How confident the model is on correct vs incorrect predictions |
| 7 | Summary Panel | Overall accuracy and F1 scores at a glance |

Pre-generated evaluation plots are included in `models/eval_plots/`.

---

## 📷 ESP32-CAM Integration

### Hardware Required

| Component | Details |
|-----------|---------|
| ESP32-CAM (AI-Thinker) | With OV2640 camera module |
| FTDI USB-to-Serial adapter | For uploading firmware |
| 5V / 2A power supply | Do NOT use 3.3V |
| Jumper wires | For GPIO0 → GND during upload |

### Uploading the Firmware

1. Open `esp32cam/FreshAgent_ESP32CAM.ino` in **Arduino IDE**.
2. Install the ESP32 board package:
   - `File → Preferences → Additional Board Manager URLs:`
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Select board: `Tools → Board → AI Thinker ESP32-CAM`
4. Select partition: `Tools → Partition Scheme → Huge APP (3MB No OTA)`
5. Edit these lines in the `.ino` file with your WiFi and server details:
   ```cpp
   const char* WIFI_SSID     = "YourWiFiName";
   const char* WIFI_PASSWORD = "YourWiFiPassword";
   const char* SERVER_HOST   = "192.168.1.XXX";  // Your PC's IP address
   const int   SERVER_PORT   = 9090;
   ```
6. Connect FTDI adapter, short GPIO0 to GND, and upload.
7. Remove the GPIO0-GND jumper and press RESET.

### How It Works

- The ESP32-CAM captures a JPEG image every 3 seconds.
- It sends the image via HTTP POST to `http://<server-ip>:9090/api/esp32-stream`.
- The server runs inference and stores the result.
- The web dashboard polls `/api/latest-esp32` to display live results.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web Dashboard |
| `GET` | `/phone-cam` | Phone Camera Page |
| `GET` | `/api/health` | Server health check |
| `POST` | `/api/upload-manual` | Upload fruit image (browser) |
| `POST` | `/api/upload-veggie` | Upload vegetable image (browser) |
| `POST` | `/api/esp32-stream` | Receive image from ESP32-CAM |
| `GET` | `/api/latest-esp32` | Poll latest ESP32 result |
| `POST` | `/api/phone-stream` | Receive frame from phone camera |
| `GET` | `/api/latest-phone` | Poll latest phone camera result |
| `GET` | `/api/server-info` | Get server LAN IP and port |
| `GET` | `/api/qr-code` | Generate QR code image for phone pairing |
| `GET` | `/api/veggie-status` | Check if VeggieAgent model is loaded |

### Example API Response (`/api/upload-manual`)

```json
{
  "fruit": "Apple",
  "condition": "Fresh",
  "fruit_confidence": 97.42,
  "cond_confidence": 99.18,
  "fruit_probs": {
    "Apple": 97.42,
    "Banana": 0.85,
    "Grape": 0.32,
    "Mango": 0.91,
    "Orange": 0.50
  },
  "cond_probs": {
    "Formalin_Mixed": 0.12,
    "Fresh": 99.18,
    "Rotten": 0.70
  },
  "safety": "SAFE - Fresh fruit. No adulteration detected.",
  "safety_class": "safe-class",
  "gradcam": "data:image/png;base64,iVBOR...",
  "timestamp": "2026-05-06T13:00:00.000000",
  "source": "manual_upload",
  "filename": "apple_test.jpg"
}
```

---

## 🍎 Supported Classes

### Fruits (FreshAgent Model)

| Fruit | Fresh | Rotten | Formalin-Mixed |
|-------|:-----:|:------:|:--------------:|
| Apple | ✅ | ✅ | ✅ |
| Banana | ✅ | ✅ | ✅ |
| Grape | ✅ | ✅ | ✅ |
| Mango | ✅ | ✅ | ✅ |
| Orange | ✅ | ✅ | ✅ |

### Vegetables (VeggieAgent Model)

| Vegetable | Fresh | Rotten | Adulterated |
|-----------|:-----:|:------:|:-----------:|
| Ginger | ✅ | ✅ | ✅ |

---

## 📂 Dataset Structure

The training dataset should follow this exact directory structure:

```
Dataset_Root/
├── Apple/
│   ├── Fresh/              ← .jpg / .png images
│   ├── Rotten/
│   └── Formalin_Mixed/
├── Banana/
│   ├── Fresh/
│   ├── Rotten/
│   └── Formalin_Mixed/
├── Grape/
│   └── ...
├── Mango/
│   └── ...
└── Orange/
    └── ...
```

> **Note:** The dataset is NOT included in this submission due to its large size. Contact the team for access if needed.

The dataset is automatically split into **70% train / 15% validation / 15% test** using stratified sampling.

---

## 📸 Screenshots

Once the server is running at `http://localhost:9090`, you will see:
- A modern dashboard with drag-and-drop image upload
- Real-time prediction results with confidence bars
- Grad-CAM heatmap visualization
- Safety verdict (SAFE / UNSAFE / DANGER)
- Live ESP32 / Phone camera feed panel with QR code

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | PyTorch, EfficientNetV2-B0 (via `timm`), Focal Loss |
| **Inference** | ONNX Runtime (CPU-optimized) |
| **Explainability** | Grad-CAM (`pytorch-grad-cam`) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML5 + CSS3 + Vanilla JavaScript |
| **Data Augmentation** | Albumentations (RandAugment, CutMix), MixUp |
| **Evaluation** | scikit-learn, matplotlib, seaborn |
| **Hardware** | ESP32-CAM (AI-Thinker) with OV2640 |
| **IoT Protocol** | HTTP POST (multipart/form-data) |

---

##  Team & Acknowledgements

| Name | Roll Number | Role |
|------|-------------|------|
| **K Guru Charan** | RA2211026010141 | Project Lead |
| **Sreenivas Nithin** | RA2211026010145 | Team Member |

**Guide:** Dr. M Meenakshi

**Institution:** SRM Institute of Science and Technology

---

## ⚠️ Disclaimer

This system detects **visual proxies** of adulteration (surface texture, color, gloss changes caused by formalin treatment). It does **NOT** perform chemical analysis.

- **Do NOT** use this system for regulatory or food safety certification purposes without laboratory validation.
- **Current scope:** Formalin detection only.
- **Does NOT detect:** Wax coatings, calcium carbide, or artificial colors.

---

## 📝 License

This project was developed for academic purposes. All rights reserved by the project team.

---

*Last updated: May 2026*

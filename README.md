# 🍎 FreshAgent — AI-Based Fruits & Vegetables Adulteration Detection System

> **A deep learning-powered food safety system** that detects whether fruits and vegetables are **Fresh**, **Rotten**, or **Formalin-treated** using EfficientNetV2 and real-time inference via an ESP32-CAM or phone camera.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Supported Items](#-supported-items)
4. [Hardware Requirements](#-hardware-requirements)
5. [Project Structure](#-project-structure)
6. [Setup Instructions](#-setup-instructions)
7. [Running the Application](#-running-the-application)
8. [Training the Models](#-training-the-models)
9. [API Documentation](#-api-documentation)
10. [ESP32-CAM Integration](#-esp32-cam-integration)
11. [Phone Camera Integration](#-phone-camera-integration)
12. [Model Performance](#-model-performance)
13. [Technical Details](#-technical-details)
14. [Troubleshooting](#-troubleshooting)
15. [Future Scope](#-future-scope)
16. [Team & Acknowledgements](#-team--acknowledgements)

---

## 🎯 Project Overview

**FreshAgent** is an AI-powered food safety system designed to detect adulteration in fruits and vegetables using deep learning. The system uses:

- **EfficientNetV2-B0** as the backbone architecture (pretrained on ImageNet)
- **Multi-task learning** — simultaneously identifies the item type AND its condition
- **ONNX Runtime** for optimized, production-grade inference
- **FastAPI** web backend for real-time analysis
- **ESP32-CAM** hardware integration for embedded camera feeds
- **Phone Camera** support for mobile-based scanning via LAN

### Key Features

- 🔍 **Dual Model System** — Separate fruit and vegetable classification models
- 📱 **Phone Camera** — Use your phone as a wireless camera via QR code pairing
- 📷 **ESP32-CAM** — Embedded camera module for standalone operation
- 🖥️ **Web Dashboard** — Beautiful real-time analysis interface
- ⚡ **Fast Inference** — < 200ms per image via ONNX Runtime
- 🛡️ **Confidence Gate** — Rejects non-food objects to prevent false positives
- 📊 **Grad-CAM** — Visual explanations showing which regions the AI focused on

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                              │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
│   │ ESP32-CAM│    │ Phone    │    │ Manual Upload    │      │
│   │ Module   │    │ Camera   │    │ (Browser)        │      │
│   └────┬─────┘    └────┬─────┘    └────────┬─────────┘      │
│        │               │                   │                 │
│        └───────────────┼───────────────────┘                 │
│                        ▼                                     │
│            ┌───────────────────────┐                         │
│            │   FastAPI Backend     │                         │
│            │   (Python Server)     │                         │
│            └─────────┬─────────────┘                         │
│                      │                                       │
│           ┌──────────┴──────────┐                            │
│           ▼                     ▼                            │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ FreshAgent Model│  │ VeggieAgent     │                   │
│  │ (Fruits - ONNX) │  │ (Veggies - ONNX)│                   │
│  │ 5 Fruits        │  │ 7 Vegetables    │                   │
│  │ 3 Conditions    │  │ 2 Conditions    │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           └──────────┬─────────┘                             │
│                      ▼                                       │
│            ┌───────────────────────┐                         │
│            │  Web Dashboard (UI)   │                         │
│            │  Results + Grad-CAM   │                         │
│            └───────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🍏 Supported Items

### Fruits (FreshAgent Model)
| Fruit  | Fresh | Rotten | Formalin-Mixed |
|--------|:-----:|:------:|:--------------:|
| 🍎 Apple  | ✅ | ✅ | ✅ |
| 🍌 Banana | ✅ | ✅ | ✅ |
| 🥭 Mango  | ✅ | ✅ | ✅ |
| 🍊 Orange | ✅ | ✅ | ✅ |
| 🍇 Grapes | ✅ | ✅ | ✅ |

### Vegetables (VeggieAgent Model)
| Vegetable      | Fresh | Rotten |
|----------------|:-----:|:------:|
| 🥬 Bitter Gourd | ✅ | ✅ |
| 🫑 Capsicum     | ✅ | ✅ |
| 🥒 Cucumber     | ✅ | ✅ |
| 🫚 Ginger       | ✅ | ✅ |
| 🌿 Okra         | ✅ | ✅ |
| 🥔 Potato       | ✅ | ✅ |
| 🍅 Tomato       | ✅ | ✅ |

---

## 💻 Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **CPU**   | Intel Core i7 13th Gen H-Series (or better) |
| **GPU**   | Intel Iris Xe (via `torch-directml`) or NVIDIA CUDA |
| **RAM**   | 8 GB minimum, 16 GB recommended |
| **Storage** | 5 GB free (model + dataset) |
| **ESP32-CAM** | AI-Thinker ESP32-CAM module (optional) |
| **Phone** | Any smartphone with a camera + same WiFi network (optional) |

---

## 📁 Project Structure

```
FreshAgent-Submission/
│
├── README.md                    ← You are here
│
├── backend/
│   ├── main.py                  # FastAPI server (all endpoints)
│   ├── inference.py             # Fruit ONNX inference engine + Grad-CAM
│   └── veggie_inference.py      # Vegetable ONNX inference engine + Grad-CAM
│
├── frontend/
│   ├── index.html               # Main web dashboard
│   ├── style.css                # All styling (dark mode, glassmorphism)
│   ├── app.js                   # JavaScript (upload, polling, rendering)
│   └── phone_cam.html           # Phone camera capture page
│
├── training/
│   ├── dataset.py               # Fruit dataset loading + stratified split
│   ├── model.py                 # EfficientNetV2-B0 multi-task architecture
│   ├── train.py                 # Training loop (2-phase, MixUp, early stop)
│   ├── export.py                # Export PyTorch → ONNX
│   └── plot_results.py          # Generate confusion matrices + curves
│
├── veggie_training/
│   ├── veggie_dataset.py        # Vegetable dataset loading (7 veggies)
│   ├── veggie_model.py          # Veggie EfficientNetV2-B0 architecture
│   ├── train.py                 # Veggie training loop
│   └── export.py                # Veggie ONNX export
│
├── esp32cam/
│   └── FreshAgent_ESP32CAM.ino  # Complete Arduino firmware for ESP32-CAM
│
├── models/
│   ├── freshagent.onnx          # Fruit model (ONNX, ~25 MB)
│   ├── veggieagent.onnx         # Veggie model (ONNX, ~25 MB)
│   ├── best_model.pth           # Fruit PyTorch checkpoint
│   ├── veggieagent_best.pth     # Veggie PyTorch checkpoint
│   └── eval_plots/              # Training evaluation plots (PNG)
│
├── requirements.txt             # Python dependencies
├── start_server.bat             # One-click server launch (Windows)
├── train.bat                    # One-click fruit training (Windows)
├── train_veggie.bat             # One-click veggie training (Windows)
├── train_overnight.bat          # Long training with auto-shutdown
├── auto_stop.py                 # Auto-stop training on convergence
└── evaluate.py                  # Evaluation script
```

---

## ⚙️ Setup Instructions

### Prerequisites

- **Python 3.10+** installed ([python.org](https://www.python.org/downloads/))
- **Git** (optional, for cloning)
- **Arduino IDE** (only if using ESP32-CAM)

### Step 1: Create Virtual Environment

```powershell
# Navigate to the project folder
cd FreshAgent-Submission

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### One-Click Setup (Alternative)

Just double-click `start_server.bat` — it will create the venv, install packages, and start the server automatically.

---

## 🚀 Running the Application

### Option A: One-Click Launch

```bat
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
| `http://localhost:9090` | 🖥️ Main Dashboard |
| `http://localhost:9090/docs` | 📖 Interactive API Docs (Swagger) |
| `http://localhost:9090/phone-cam` | 📱 Phone Camera Page |

### Using the Dashboard

1. **Select Mode** — Toggle between "Fruit Detection" and "Vegetable Detection"
2. **Upload an Image** — Drag & drop or click to upload a fruit/vegetable photo
3. **View Results** — See the AI prediction with safety label and confidence score
4. **Grad-CAM** — View the heatmap showing which regions the AI focused on

---

## 🧠 Training the Models

### Fruit Model Training

```powershell
# Using batch script
train.bat "C:\path\to\fruit\dataset"

# Manual
venv\Scripts\activate
python training/train.py --data_dir "C:\path\to\dataset" --epochs 60 --batch_size 16
python training/export.py --checkpoint models/best_model.pth --output models/freshagent.onnx
```

### Vegetable Model Training

```powershell
# Using batch script
train_veggie.bat

# Manual
venv\Scripts\activate
python veggie_training/train.py --data_dir "Augmented-Resized Image" --output_dir models --epochs 5 --batch_size 16
python veggie_training/export.py --checkpoint models/veggieagent_best.pth --output models/veggieagent.onnx
```

### Training Time Estimates (Intel Iris Xe, batch_size=16)

| Model | Epochs | Dataset Size | Estimated Time |
|-------|--------|-------------|----------------|
| Fruit | 60 | ~7,500 images | ~90–150 min |
| Veggie | 5 | ~48,000 images | ~2–3 hours |

> **Tip:** Use `--batch_size 8` if you run out of memory.

### Dataset Structure

**Fruit Dataset:**
```
dataset_root/
├── Apple/
│   ├── Fresh/          ← .jpg, .png images
│   ├── Rotten/
│   └── Formalin_Mixed/
├── Banana/ ...
└── Grapes/ ...
```

**Vegetable Dataset:**
```
dataset_root/
├── dataset/
│   ├── Train/
│   │   ├── freshbittergourd/
│   │   ├── rottenbittergourd/
│   │   ├── freshcapsicum/
│   │   └── ...
│   └── Test/ ...
└── Ginger/
    ├── train/
    │   ├── Fresh/
    │   └── Adulterated/    ← mapped to Rotten
    └── test/ ...
```

---

## 📡 API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/api/health` | Server health check + fruit model status |
| `GET` | `/api/veggie-status` | Vegetable model status |
| `POST` | `/api/upload-manual` | Upload image for fruit prediction |
| `POST` | `/api/upload-veggie` | Upload image for veggie prediction |
| `POST` | `/api/esp32-stream` | Receive ESP32-CAM image stream |
| `GET` | `/api/latest-esp32` | Poll latest ESP32 prediction result |
| `POST` | `/api/phone-frame` | Receive phone camera frame |
| `GET` | `/api/latest-phone` | Poll latest phone camera result |
| `GET` | `/api/qr-code` | Generate QR code for phone pairing |
| `GET` | `/phone-cam` | Phone camera page |

### Example: Upload an Image

```bash
curl -X POST "http://localhost:9090/api/upload-manual" \
  -F "file=@apple.jpg"
```

### Example Response

```json
{
  "fruit": "Apple",
  "fruit_confidence": 99.2,
  "condition": "Fresh",
  "condition_confidence": 97.8,
  "safety": "SAFE - Fresh fruit. No adulteration detected.",
  "safety_class": "safe-class",
  "gradcam_b64": "data:image/jpeg;base64,..."
}
```

---

## 📷 ESP32-CAM Integration

### Hardware Setup

1. **ESP32-CAM Module** — AI-Thinker ESP32-CAM
2. **FTDI Programmer** — For flashing firmware
3. **5V/2A Power Supply** — Essential for stable operation (USB may cause brownout)
4. **10µF Capacitor** — Between 5V and GND to prevent brownout resets

### Firmware Upload

1. Open `esp32cam/FreshAgent_ESP32CAM.ino` in Arduino IDE
2. Set your WiFi credentials in the code:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Set the server IP:
   ```cpp
   const char* serverUrl = "http://YOUR_SERVER_IP:9090/api/esp32-stream";
   ```
4. Select Board: **AI Thinker ESP32-CAM**
5. Upload and monitor via Serial (115200 baud)

### How It Works

1. ESP32-CAM captures a JPEG image every 5 seconds
2. Image is sent via HTTP POST to the backend
3. Backend runs AI inference and stores the result
4. Dashboard polls `/api/latest-esp32` and displays the result

---

## 📱 Phone Camera Integration

### How to Use

1. Start the server on your laptop/PC
2. Open the dashboard at `http://localhost:9090`
3. Click "📱 Phone Camera" button on the dashboard
4. Scan the QR code with your phone (both devices must be on same WiFi)
5. On your phone, tap to capture images
6. Select "Fruit" or "Veggie" mode on the phone page
7. Results appear on both the phone and the main dashboard

> **Note:** The phone camera page works over HTTP on your local network. Some browsers may require you to manually allow camera access.

---

## 📊 Model Performance

### FreshAgent (Fruit Model)

| Metric | Score |
|--------|:-----:|
| **Fruit Identification F1** | 99.5% |
| **Condition F1 (Fresh/Rotten/Formalin)** | 96.8% |
| **ONNX Model Size** | 25 MB |

### VeggieAgent (Vegetable Model)

| Metric | Score |
|--------|:-----:|
| **Vegetable Identification F1** | 99.95% |
| **Condition F1 (Fresh/Rotten)** | 97.00% |
| **ONNX Model Size** | 25 MB |
| **Training Dataset** | 48,532 images |

---

## 🔬 Technical Details

### Deep Learning Pipeline

```
Image Input (224×224 RGB)
    ↓
EfficientNetV2-B0 Backbone (ImageNet pretrained)
    ↓
Shared Feature Extraction (1280-d vector)
    ↓
┌──────────────────┬──────────────────┐
│ Item Head        │ Condition Head   │
│ (5 fruits OR     │ (Fresh/Rotten/   │
│  7 vegetables)   │  Formalin)       │
└──────────────────┴──────────────────┘
    ↓
Multi-Task Prediction
    ↓
Confidence Gate (75% threshold)
    ↓
Safety Label + Grad-CAM Heatmap
```

### Key Technical Choices

| Choice | Rationale |
|--------|-----------|
| **EfficientNetV2-B0** | Best accuracy/speed trade-off for edge deployment |
| **Multi-Task Learning** | Single forward pass identifies both item type and condition |
| **ONNX Runtime** | 3-5x faster than PyTorch for inference |
| **Focal Loss** | Handles class imbalance in condition labels |
| **MixUp Augmentation** | Improves generalization and reduces overfitting |
| **2-Phase Training** | Frozen backbone warmup → full fine-tuning |
| **Confidence Gate** | Prevents false positives on non-food objects |
| **DirectML** | Enables GPU training on Intel Iris Xe |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Server won't start** | Ensure `venv` is activated and dependencies installed |
| **Model not loaded** | Check that `.onnx` files exist in `models/` folder |
| **ESP32-CAM brownout** | Use 5V/2A power supply + 10µF capacitor |
| **Phone camera not working** | Ensure both devices are on the same WiFi network |
| **Low confidence predictions** | Ensure good lighting and the item fills the frame |
| **CUDA/DirectML errors** | Fall back to CPU with `--device cpu` flag |
| **Port 9090 already in use** | Kill existing process or change port in start_server.bat |

---

## 🔮 Future Scope

- Additional adulterant types (wax coating, calcium carbide detection)
- Multi-language support for wider accessibility
- Mobile app (React Native / Flutter)
- Cloud deployment (AWS/GCP) for remote access
- Integration with FSSAI compliance reporting
- Expanded vegetable support (leafy greens, root vegetables)
- Batch processing for supply chain inspection

---

## 📝 Important Disclaimer

This system detects **visual proxies** of adulteration (surface texture, color, gloss changes). It does **not** perform chemical analysis. Do not use for regulatory or food safety certification purposes without laboratory validation.

**Current scope:** Formalin detection (fruits) and freshness detection (vegetables).
**Does NOT detect:** Wax, calcium carbide, or artificial colors.

---

## 👥 Team & Acknowledgements

| Name | Roll Number | Role |
|------|-------------|------|
| **K Guru Charan** | RA2211026010141 | Project Lead |
| **Sreenivas Nithin** | RA2211026010145 | Team Member |

**Guide:** Dr. M Meenakshi

**Institution:** SRM Institute of Science and Technology

---

## 📜 License

This project was developed as part of an academic project at SRM Institute of Science and Technology. All rights reserved.

---

*Built with ❤️ using Python, FastAPI, EfficientNetV2, and ONNX Runtime*

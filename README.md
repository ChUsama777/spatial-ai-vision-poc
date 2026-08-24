<div align="center">

# 🌌 Spatial AI Vision PoC
**Automated Dimensional Measurement & 3D Object Localization**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v26-00FFFF?logo=ai&logoColor=black)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)]()
[![Open3D](https://img.shields.io/badge/Open3D-Point%20Clouds-blue)]()

</div>

---

## 📖 Overview
This repository contains the Proof of Concept (PoC) for a **Spatial AI and Computer Vision Architecture** designed to solve high-value enterprise manufacturing problems. The primary objective of this project is to automate the detection of millimeter/centimeter tolerance errors in high-end 3D-printed manufacturing parts (e.g., aerospace and automotive parts).

By leveraging state-of-the-art Object Detection models, Monocular Depth Estimation, and Point Cloud processing, this pipeline eliminates the need for expensive hardware scanners by computing spatial dimensions (L x W x H) directly from 2D images and generated depth maps.

---

## ✨ Key Features & Benchmarks

### 1. 🎯 YOLO Architecture Benchmarking (YOLOv8 vs. YOLO26)
- **Direct Comparison:** Head-to-head benchmarking between legacy `YOLOv8` and the latest NMS-free end-to-end `YOLO26` architecture.
- **Out-of-Vocabulary Testing:** Evaluated on non-COCO 3D objects (Tissue Box, HBL Credit Card, Mobile Phones) to test spatial boundary logic and hallucination handling.
- **Results:** YOLO26 demonstrated a ~15% faster inference time with near-zero post-processing latency, validating its NMS-free head.

### 2. 🕳️ Monocular Depth Estimation
- Integrated **Depth-Anything-V2** and **Metric3D** to generate highly accurate depth maps from single RGB visual frames.
- Applied customized inferencing using OpenCV and PyTorch to visualize normalized depth outputs (Inferno Colormap).

### 3. 📐 Point Cloud Processing & Dimensional Extraction
- Developed a 3D geometry processing pipeline using **Open3D**.
- **Workflow:** `Depth Map` ➔ `3D Point Cloud` ➔ `Bounding Box Extraction` ➔ `Metric Dimension Measurement`.

### 4. 📷 Photogrammetry Baseline Generation
- Integrated **AliceVision Meshroom** workflows to process 2D image datasets into highly accurate 3D meshes.
- Acts as the baseline (ground truth) for comparing newly manufactured 3D-printed parts to detect structural anomalies.

---

## 🛠️ Tech Stack
*   **Computer Vision:** OpenCV, Ultralytics (YOLO)
*   **Deep Learning & AI:** PyTorch, Depth-Anything-V2, Metric3D
*   **3D Geometry:** Open3D, AliceVision Meshroom
*   **Language:** Python 3.10+

---

## 📂 Repository Structure
```text
spatial-ai-vision-poc/
│
├── YOLO_Benchmarking/         # YOLOv8 vs YOLO26 comparison scripts
├── Depth_Estimation/          # Depth-Anything-V2 & Metric3D inferencing code
├── 3D_Reconstruction/         # Meshroom photogrammetry pipeline scripts
├── PointCloud_Processing/     # Open3D dimension extraction logic
├── sample_images/             # Test objects (Tissue box, Cards, etc.)
└── README.md

```

---

## ⚙️ Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed. A CUDA-enabled GPU is highly recommended for faster inference.

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/ChUsama777/spatial-ai-vision-poc.git](https://github.com/ChUsama777/spatial-ai-vision-poc.git)
cd spatial-ai-vision-poc
pip install torch torchvision torchaudio
pip install opencv-python ultralytics open3d numpy

```

*(Note: For Depth-Anything-V2, ensure you download the respective `.pth` weights into the `checkpoints` directory).*

### 3. Running the Pipeline

**To run YOLO Benchmarks:**

```bash
python YOLO_Benchmarking/compare_models.py

```

**To generate a Depth Map:**

```bash
python Depth_Estimation/test_dept.py

```

---

## 👤 Author

**Usama Ashraf**

*AI Engineer & Spatial Computing Enthusiast*

[LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/ChUsama777) | [GitHub](https://github.com/ChUsama777)

---

*Note: This PoC was originally developed during a B2B Spatial AI R&D sprint to address hardware tolerance verification.*

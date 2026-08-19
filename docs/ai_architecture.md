# ITMS — AI Pipeline Architecture Specification

## 1. Executive Summary & Strategy

The ITMS prototype adopts an **Edge-Optimized Dual-Track Hybrid AI Architecture** combined with a **Multimodal Sensor Fusion Engine**.

Rather than relying on a single monolithic deep learning network for every task, the pipeline combines:
1. **Supervised Deep Learning (YOLOv8n)** for high-confidence foreign obstacle and hazard detection.
2. **Deterministic & Classical Computer Vision** for vegetation segmentation and dual-rail geometric tracking (zero hallucination, highly explainable).
3. **Unsupervised Anomaly Scoring** for structural sleeper and ballast texture deviations.
4. **Multimodal Telemetry Fusion** combining vision alerts with replayed IMU vibration spikes and GPS coordinates into a composite **Track Health Index (THI)**.

```
                  ┌──────────────────────────────────────────────┐
                  │    Raw Train Front-Facing POV Video Stream   │
                  │             (1080p, 30 FPS)                  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Deterministic Track Bed ROI Extraction     │
                  │     (1920x1080 -> 1612x572, 55% discarded)   │
                  └───────┬──────────────┬──────────────┬────────┘
                          │              │              │
        ┌─────────────────┴─┐     ┌──────┴───────┐   ┌──┴──────────────────┐
        │ Supervised ML     │     │ Classical CV │   │ Unsupervised        │
        │ Track Obstacle    │     │ Vegetation   │   │ Sleeper / Ballast   │
        │ Detector (YOLOv8n)│     │ & Rail Track │   │ Texture Anomaly     │
        └─────────┬─────────┘     └──────┬───────┘   └──┬──────────────────┘
                  │                      │              │
                  │   Visual Detection   │              │
                  │   & Geometry Scores  │              │
                  └───────────────┬──────┴──────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │               MULTIMODAL FUSION ENGINE                 │
      │                                                        │
      │  [Vision Score]  +  [Replayed IMU]  +  [Replayed GPS]  │
      │   (Obstacles/Veg)   (Vibration Gz)     (Chainage km)   │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │       Track Health Index (THI) & Alert Generator       │
      │       THI: 0 - 100  |  NORMAL/ADVISORY/WARNING/CRITICAL │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │            FastAPI REST & WebSocket Backend            │
      │                           +                            │
      │             React + Vite Frontend Dashboard            │
      └────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Components

### Module 1: Deterministic Track ROI Extraction ([ai_pipeline/detection/track_roi.py](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/ai_pipeline/detection/track_roi.py))
- Crops the active track corridor ($y: 486 - 1058, x: 154 - 1766$).
- Strips sky, clouds, overhead wires, and locomotive edge.
- Discards $55.5\%$ unnecessary background area, doubling feature extraction density.

### Module 2: Supervised Hazard Detector (`YOLOv8n-ITMS`)
- **Model Choice**: Ultralytics **YOLOv8n (Nano)**.
- **Model Size**: ~3.2M parameters (~6.4 MB model weights).
- **Target Classes**: `track_obstruction`, `rock_debris`, `fallen_branch`, `human_trespasser`, `animal`, `hazard`.
- **Inference Input**: $640 \times 384$ cropped ROI tensor.
- **Hardware Budget**: Runs at **$45 - 60\text{ FPS}$** on an NVIDIA RTX 3050 Laptop GPU (consuming $<1.2\text{ GB}$ VRAM).
- **Rationale**: Extreme inference efficiency, zero latency bottleneck, easily trained locally or on Google Colab in $<20\text{ minutes}$.

### Module 3: Explainable Classical Vision Modules
- **Vegetation Encroachment**: Excess Green Index ($\text{ExG} = 2G - R - B$) with morphological filtering and track corridor polygon masking. Zero training required, $100\%$ explainable.
- **Dual-Rail Geometry & Visual Alignment**: Sobel-X gradient filtering and 2nd-order polynomial spline fitting ($x = ay^2 + by + c$) to compute visual gauge variation and track curvature deviation.

### Module 4: Unsupervised Anomaly Scoring
- **Sleeper Pitch Regularity**: Fast Fourier Transform / Autocorrelation spatial peak detection across longitudinal scanlines to identify irregular sleeper spacing or transverse structural breaks.
- **Ballast Void Anomaly**: Gray-Level Co-occurrence Matrix (GLCM) texture entropy tracking to highlight ballast scouring and uneven bed distribution.

### Module 5: Multimodal Sensor Fusion & Track Health Index (THI)
- **Sensor Replay Interfaces**:
  - `IMUTelemetry`: Time-series vertical acceleration ($G_z$), lateral acceleration ($G_y$), and angular velocity ($\omega_z$).
  - `GPSLocation`: Latitude, Longitude, Altitude, Speed ($km/h$), and Track Chainage ($km$).
  - `LiDARProfile`: Simulated railhead clearance and ballast height profile.
- **Fusion Formula**:
  $$\text{Severity}_{\text{composite}} = \max\left(w_{\text{vis}} S_{\text{vis}},\, w_{\text{imu}} S_{\text{imu}},\, w_{\text{geo}} S_{\text{geo}}\right)$$
  $$\text{Track Health Index (THI)} = 100 \times \left(1 - \text{Severity}_{\text{composite}}\right)$$

---

## 3. Alert Classification & Thresholds

| THI Range | Alert Status | Color Code | Action Required |
| :--- | :--- | :--- | :--- |
| **85 – 100** | `NORMAL` | Green (`#22C55E`) | Routine monitoring; track parameter nominal. |
| **70 – 84** | `ADVISORY` | Yellow (`#EAB308`) | Minor vegetation or surface anomaly detected; scheduled maintenance. |
| **50 – 69** | `WARNING` | Orange (`#F97316`) | Track geometry irregularity or sleeper anomaly; caution order recommended. |
| **0 – 49** | `CRITICAL` | Red (`#EF4444`) | Severe foreign obstruction or major vibration shock; immediate stop / emergency alert. |

---

## 4. Hardware Optimization & Feasibility (NVIDIA RTX 3050)

- **Available GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 13.0, Driver 581.86).
- **VRAM Allocation Budget**:
  - PyTorch + YOLOv8n runtime: $\sim 1.0\text{ GB}$
  - Frame buffers & OpenCV pipelines: $\sim 0.5\text{ GB}$
  - Peak Total Usage: $\le 1.8\text{ GB}$ ($< 30\%$ of total 6GB capacity).
- **Expected Throughput**: $> 45\text{ FPS}$ on batch size 1 (real-time capable for 30 FPS video).

---

## 5. Scope & Phasing (MVP vs Optional)

### MVP Deliverables (SIH Prototype Core):
1. [x] Video ingestion & Frame extraction at 6 FPS ([ai_pipeline/detection/frame_extractor.py](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/ai_pipeline/detection/frame_extractor.py)).
2. [x] Deterministic Track Bed ROI Extractor ([ai_pipeline/detection/track_roi.py](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/ai_pipeline/detection/track_roi.py)).
3. [ ] Supervised Obstacle Detector (YOLOv8n).
4. [ ] Algorithmic Vegetation & Dual-Rail Geometry Tracker.
5. [ ] IMU & GPS Telemetry Replay Engine.
6. [ ] Multimodal Fusion Service & THI Alert Dispatcher.
7. [ ] FastAPI Backend + React/Vite Dashboard.

### Optional / Future Extensions:
- Near-field missing fastener clip classifier.
- Real-time 3D LiDAR point cloud point-mesh rendering.
- PostgreSQL database migration.

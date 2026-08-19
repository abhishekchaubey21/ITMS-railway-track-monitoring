# ITMS — Dataset Strategy & Governance Document

## 1. Overview & Core Data Principles

In strict alignment with the development principles established in [AGENTS.md](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/AGENTS.md), the ITMS prototype enforces absolute clarity and honesty regarding data provenance:

1. **Clear Categorical Distinction**:
   - **Real Recorded Video**: Primary real-world front-facing train POV footage (`data/raw/train_pov.mp4`).
   - **Public Benchmark Datasets**: Curated open-access datasets used for model weights/training reference (e.g. RailSem19, Rail Track Obstacle Dataset).
   - **Simulated / Synthetic Data**: Explicitly created synthetic hazard injections used for reproducible stress-testing (clearly marked as `simulated/`).
   - **Replayed Sensor Telemetry**: Calibrated time-series sensor logs (IMU vibration, GPS chainage, LiDAR profiles) replayed to demonstrate multimodal fusion.
2. **Zero Fabrication**: Simulated or replayed sensor telemetry will **never** be presented as real-time synchronized measurements from the same train run unless actual physical synchronization was recorded.
3. **Controlled Downloads**: Datasets and model weights are tracked through metadata manifests ([data/dataset_registry.json](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/data/dataset_registry.json)) before any downloads occur.

---

## 2. Primary Video Baseline

| Property | Value |
| :--- | :--- |
| **Filename** | `data/raw/train_pov.mp4` |
| **Camera Viewpoint** | Front-facing cab windshield train POV |
| **Resolution** | $1920 \times 1080$ (Full HD 1080p) |
| **FPS / Duration** | $30.0\text{ FPS}$ / $184.33\text{ seconds}$ ($5,530\text{ total frames}$) |
| **Extracted Working Set** | $1,106\text{ frames}$ ($1\text{ frame every } 5\text{ source frames} \approx 6.0\text{ FPS}$) |
| **Track Bed ROI** | $1612 \times 572\text{ px}$ ($y: 486 - 1058, x: 154 - 1766$) |
| **Integrity & Quality** | 100% readable, zero corrupt frames, high sharpness (Mean Lapl. Var: $504.6$) |

---

## 3. Public Candidate Datasets Survey

### A. RailSem19
- **Source**: [https://wilddash.cc/railsem19](https://wilddash.cc/railsem19) / CVPRW 2019
- **License**: Creative Commons BY-NC-SA 4.0 (Approved for non-commercial research & hackathons)
- **Modality & Size**: 8,500 Full HD train/tram forward POV images (~12 GB)
- **Annotation Format**: Polygons, Polylines, Dense Semantic Masks
- **Classes**: Rails, track bed, switches, signals, trains, buffer stops, crossings
- **Relevance**: **95% (Highest)**. Direct match for cab-view rail line perception and track corridor segmentation.

### B. Rail Track Obstacle Dataset (Roboflow Universe)
- **Source**: [Roboflow Universe — Rail Obstacle Detection](https://universe.roboflow.com/rail-obstacle-detection/dataset-for-detecting-obstacles-on-railways)
- **License**: CC BY 4.0 / Public Domain
- **Modality & Size**: ~700-1,200 annotated images (~350 MB)
- **Annotation Format**: YOLO format bounding boxes (`.txt`)
- **Classes**: `fallen_tree`, `rock_debris`, `human_trespasser`, `animal`, `vehicle_on_track`, `track_obstruction`
- **Relevance**: **90%**. Perfect target alignment for foreign obstacle and collision hazard detection.

### C. OSDaR23 (Open Sensor Data for Rail 2023)
- **Source**: [https://fid-move.de/data-sets/osdar23](https://fid-move.de/data-sets/osdar23) / German Federal Ministry / ArXiv 2023
- **License**: CC BY-SA 4.0 (Open academic research)
- **Modality**: Synchronized IMU accelerometer ($G_x, G_y, G_z$), GNSS/GPS, LiDAR point clouds, and RGB cameras.
- **Relevance**: **92%**. Provides the exact physical schema and real-world variance distribution for our IMU vibration and GPS telemetry replay service.

### D. RFDD (Railway Fastener Defect Dataset)
- **Source**: [Roboflow Universe — Fastener Detection](https://universe.roboflow.com/railway-defects-osolv/fastener-detection-7a3vn)
- **License**: CC BY 4.0
- **Modality & Size**: ~500 close-up images (~200 MB)
- **Classes**: `good_fastener`, `missing_fastener`, `defective_clip`
- **Relevance**: **65% (Secondary)**. Downward angle differs from cab POV, but applicable for localized near-field patch testing.

### E. RSDDs (Rail Surface Defect Datasets - Type I & II)
- **Source**: [NEU Rail Surface Defect Dataset](https://github.com/neu-rail-rsdds/rsdds)
- **License**: Open Academic Use (Northeastern University)
- **Modality & Size**: 195 high-resolution grayscale railhead crops (~50 MB)
- **Classes**: Rail surface cracks, spalling, squats, burns
- **Relevance**: **50% (Secondary Reference)**. Downward nadir perspective used for understanding railhead signal disruption patterns.

---

## 4. Dataset-to-Capability Mapping

```
ITMS Capability
    ↓
Dataset(s)
    ↓
Classes Usable
    ↓
Required Preprocessing
    ↓
Expected Feasibility
```

| ITMS Capability | Source Dataset | Usable Classes / Signals | Preprocessing Required | Feasibility & AI Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Foreign Obstacle Detection** | Rail Track Obstacle Dataset | `rock`, `tree_branch`, `animal`, `human`, `debris` | Crop to Track ROI, normalize to $640\times 384$ | **HIGH** (Supervised YOLOv8n) |
| **2. Vegetation Encroachment** | Native `train_pov.mp4` | `vegetation_mask` (ExG $> 25$) | Excess Green index conversion, track corridor intersection | **HIGH** (Algorithmic CV, Zero ML training) |
| **3. Dual-Rail & Geometry Indicators** | RailSem19 + Native POV | `left_rail`, `right_rail`, `center_line` | Sobel-X edge filtering, 2nd-order polynomial curve fitting | **MEDIUM-HIGH** (Classical CV + Spline Tracking) |
| **4. Sleeper / Ballast Anomalies** | Native `train_pov.mp4` + Synthetic | `sleeper_pitch_deviation`, `texture_entropy` | Spatial frequency autocorrelation + GLCM texture entropy | **MEDIUM** (Unsupervised Patch Baseline) |
| **5. Missing Fasteners (Near-Field)** | RFDD Fastener Dataset | `missing_clip`, `intact_clip` | Localized crop at $y > 0.75 H$ rail-sleeper junction | **LOW-MEDIUM** (Lightweight Patch Classifier) |
| **6. Sensor Telemetry Replay & Fusion** | OSDaR23 Schema + Synthetic Log Generator | `accel_z` ($m/s^2$), `gps_lat_lon`, `chainage_km` | Timestamp alignment with video frame sequence | **HIGH** (Replay Engine + Rule-Based Fusion) |

---

## 5. Licensing & Hackathon Compliance

All selected candidate datasets possess permissive open-access licenses:
- **Roboflow Datasets**: CC BY 4.0 / Public Domain (Commercial & Non-Commercial compliant with attribution).
- **RailSem19**: CC BY-NC-SA 4.0 (Explicitly permits non-commercial academic research and prototype demonstrations).
- **OSDaR23**: CC BY-SA 4.0 (Open academic and technical demonstration compliant).

---

## 6. Implementation Manifest

Dataset metadata is centralized in [data/dataset_registry.json](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/data/dataset_registry.json) and queryable via the [`DatasetRegistry`](file:///c:/Users/Abhishek/Projects/ITMS-railway-track-monitoring/ai_pipeline/models/dataset_registry.py) class.

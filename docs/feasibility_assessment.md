# ITMS — Railway Track Inspection Feasibility Assessment

## 1. Executive Summary

- **Source Dataset**: `train_pov.mp4` (Train Front-Facing Cab POV)
- **Native Resolution**: `1920 × 1080` (Full HD 1080p)
- **Track ROI Crop Resolution**: `1612 × 572` ($55.5\%$ background discarded)
- **Evaluated Dataset Size**: `1106` sampled video frames

### Summary of Inspection Feasibility

| Feasibility Rating | Target Count | Description |
| :--- | :--- | :--- |
| **HIGH** | **2** | Realistically detectable with high confidence via optical vision (Foreign objects, vegetation encroachment). |
| **MEDIUM** | **3** | Feasible for macroscopic conditions (Track alignment/curvature, damaged sleepers, ballast voids). |
| **LOW** | **2** | Limited optical feasibility; near-field only (Missing fasteners, major rail burns). |
| **NOT FEASIBLE** | **2** | Cannot be resolved from optical camera alone (Internal metallurgical flaws, sub-millimeter certified geometry). |

---

## 2. Infrastructure Visibility from Front-Facing Cab POV

1. **Running Rails (Left & Right)**: Consistently visible from immediate near field ($0\text{m}$) to vanishing horizon ($>150\text{m}$). The top rail head running band provides a distinct reflective line feature.
2. **Sleepers (Concrete Ties)**: Consistently visible at periodic intervals (~$60\text{cm}$ center-to-center). Sleeper bodies span $>400\text{px}$ in the near field.
3. **Fasteners (Elastic Rail Clips / Pandrol)**: Visible in near-field bottom $25\%$ of ROI ($\sim 15 - 25\text{px}$ footprint); viewed at an oblique angle.
4. **Ballast Bed**: Clear texture distribution between and around rails across all daylight frames.
5. **Track Gauge Corridor**: Fully visible corridor between rails ideal for forward obstacle and vegetation detection.

---

## 3. Systematic Target Feasibility Matrix

### Foreign Objects & Track Obstructions — **[HIGH]**
- **Category**: Obstacle & Hazard Detection
- **Visibility**: `CONSISTENTLY_CLEAR`
- **Feasibility Rating**: `HIGH`
- **Assessment & Reasoning**: Front-facing train POV offers an unobstructed forward field of view across the track gauge. Medium-to-large foreign objects (fallen boulders, debris, branches, large metallic components, animals, trespassing hazards) occupy significant pixel area (>50-200px) and present high visual contrast against uniform ballast.
- **Recommended CV Technique**: `Supervised Object Detection (YOLO / SSD) bounded within track corridor polygon + Background Anomaly / Foreground segmentation.`
- **Requires External Labels**: `Yes`
- **Multimodal / Sensor Supplement**: Can be supplemented by simulated obstacle scenarios and emergency braking telemetry.
- **Key Limitations**: *Small objects (< 5cm) at distances > 30m may be obscured by ballast texture.*

### Vegetation Encroachment into Gauge — **[HIGH]**
- **Category**: Environmental & Corridor Hazards
- **Visibility**: `CONSISTENTLY_CLEAR`
- **Feasibility Rating**: `HIGH`
- **Assessment & Reasoning**: Weeds, overgrown shrubs, and vegetation encroaching from embankment or between sleepers have distinct color signatures (green/yellow in HSV/ExG space) and high frequency texture differing sharply from ballast gravel and steel rails.
- **Recommended CV Technique**: `Excess Green (ExG) index color-space thresholding + morphological contour filtering within the active track ROI corridor.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Correlated with GPS chainage milestones for maintenance logging.
- **Key Limitations**: *Dead/dry brown grass requires texture segmentation in addition to color filters.*

### Track Alignment & Macroscopic Geometry Anomalies — **[MEDIUM]**
- **Category**: Track Geometry & Alignment
- **Visibility**: `CONSISTENTLY_CLEAR`
- **Feasibility Rating**: `MEDIUM`
- **Assessment & Reasoning**: Both left and right running rails are continuously visible from the near field to the vanishing horizon. Macroscopic geometric distortions (> 50-100mm track shifts, track buckling, severe kinks, washouts) create visible polynomial trajectory deviations and visual gauge divergence.
- **Recommended CV Technique**: `Dual-rail edge tracking (Sobel-X + Hough line / spline polynomial fitting) + Visual Gauge Variation Index calculation along vertical scanlines.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Essential to fuse with replayed/simulated IMU lateral acceleration and LiDAR profile to confirm dynamic track geometry standards.
- **Key Limitations**: *Optical camera cannot achieve millimeter-accurate (±1mm) geometry certification required for high-speed track standards without calibrated stereo LiDAR.*

### Damaged / Broken / Misaligned Sleepers — **[MEDIUM]**
- **Category**: Sleeper & Tie Structural Integrity
- **Visibility**: `MODERATELY_VISIBLE`
- **Feasibility Rating**: `MEDIUM`
- **Assessment & Reasoning**: Concrete sleepers are visible at periodic intervals (~60cm spacing). In the near field (bottom 40% of ROI), sleeper bodies span >400px width. Significant transverse breaks, severe concrete spalling, missing sleeper ends, and angular skew (>5°) are visually distinguishable.
- **Recommended CV Technique**: `Periodic sleeper pitch tracking (spatial frequency / peak detection) + patch-based visual anomaly detection / edge discontinuity scoring.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Simulated sleeper crack and displacement datasets for validation.
- **Key Limitations**: *Hairline cracks (<2mm) or fractures obscured beneath ballast dust/shadows cannot be resolved at normal operating speeds from cab height.*

### Ballast Bed Abnormalities & Severe Voids — **[MEDIUM]**
- **Category**: Track Bed & Ballast
- **Visibility**: `MODERATELY_VISIBLE`
- **Feasibility Rating**: `MEDIUM`
- **Assessment & Reasoning**: Ballast texture and distribution between and outside rails is clearly visible. Large ballast voids (scouring under sleeper ends), waterlogging/mud pumping, or excessive ballast heaps over rail base alter local texture statistics.
- **Recommended CV Technique**: `Gray-Level Co-occurrence Matrix (GLCM) texture entropy + statistical patch variance baseline.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Fused with replayed IMU vertical vibration peaks to detect ballast voids.
- **Key Limitations**: *Fine ballast grading / particle wear cannot be measured optically.*

### Missing / Displaced Fasteners (Clips) — **[LOW]**
- **Category**: Fastener & Fastening System
- **Visibility**: `LIMITED_RESOLUTION`
- **Feasibility Rating**: `LOW`
- **Assessment & Reasoning**: Elastic rail clips (Pandrol/ERC) are visible only in the extreme near field (bottom 25% of ROI) where they occupy ~15-25px. Missing clips create a noticeable contrast gap against sleeper shoulders, making missing clip detection feasible in the near field. However, oblique frontal viewing angle causes partial perspective occlusion.
- **Recommended CV Technique**: `High-resolution localized ROI crop at rail-sleeper intersections + lightweight patch classifier / template matcher.`
- **Requires External Labels**: `Yes`
- **Multimodal / Sensor Supplement**: Synthetic missing fastener injection for pipeline verification.
- **Key Limitations**: *Fastener micro-cracks, fatigue loosening, or missing clip insulators cannot be reliably detected from a cab POV camera (requires downward-pointing bogie cameras).*

### Major Rail Surface Defects & Breaks — **[LOW]**
- **Category**: Rail Surface & Head Flaws
- **Visibility**: `LIMITED_RESOLUTION`
- **Feasibility Rating**: `LOW`
- **Assessment & Reasoning**: The top rail running surface appears as a continuous bright reflective band. Major wheel burns, large spalling, or complete transverse rail breaks interrupt this specular reflection line. However, subtle rolling contact fatigue (squats, shallow head checks, micro-flaking) is below the camera resolving limit.
- **Recommended CV Technique**: `1D longitudinal intensity profile tracking along the detected rail head reflection band.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Fused with replayed IMU vibration shocks and simulated ultrasonic anomaly flags.
- **Key Limitations**: *Shallow micro-cracks (<3mm) and fine head checking cannot be detected.*

### Internal / Subsurface Rail Flaws — **[NOT_FEASIBLE]**
- **Category**: Internal Material & Metallurgical Defects
- **Visibility**: `NOT_VISIBLE`
- **Feasibility Rating**: `NOT_FEASIBLE`
- **Assessment & Reasoning**: Internal metallurgical defects (hydrogen shatter cracks, internal transverse fissures, inclusion clusters, bolt-hole internal cracks) reside entirely beneath the steel rail surface and emit zero optical signature.
- **Recommended CV Technique**: `Not feasible via optical vision. Requires Ultrasonic (UT) or Eddy Current sensors.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Must be represented via simulated / replayed non-destructive evaluation (NDE) sensor logs in multimodal fusion layer.
- **Key Limitations**: *Physical law: optical cameras cannot penetrate solid steel.*

### Millimeter-Certified Dynamic Track Gauge & Cant — **[NOT_FEASIBLE]**
- **Category**: Track Geometry Certification
- **Visibility**: `LIMITED_RESOLUTION`
- **Feasibility Rating**: `NOT_FEASIBLE`
- **Assessment & Reasoning**: Standard railway geometry certification (e.g. RDSO/EN 13848) requires ±1.0mm dynamic gauge, twist, and cant accuracy. Single-camera cab perspective without structured laser illumination is subject to vehicle suspension pitch/roll dynamics and cannot guarantee certified millimeter tolerances.
- **Recommended CV Technique**: `Qualitative visual alignment estimation only; certified measurements require LiDAR & IMU.`
- **Requires External Labels**: `No (Unsupervised / Algorithmic)`
- **Multimodal / Sensor Supplement**: Represented via replayed LiDAR profile streams and IMU gyroscope integration.
- **Key Limitations**: *Vehicle vibration and optical perspective preclude sub-millimeter metrology.*

---

## 4. Explicit Physical & Optical Limitations

In strict accordance with **AGENTS.md** principles, the following physical inspection boundaries are established:

1. **Internal Subsurface Defects**: Flaws such as internal transverse fissures, inclusion clusters, or bolt-hole cracking inside solid steel rails emit zero optical surface signature. They cannot be detected by any camera and require Ultrasonic Testing (UT) / Eddy Current probes (represented via replayed/simulated telemetry in ITMS).
2. **Sub-Millimeter Certified Geometry**: Dynamic track geometry certification (requiring $\pm 1.0\text{mm}$ precision) cannot be certified from a vibrating cab-mounted optical camera without calibrated LiDAR and gyro-compensated IMU sensors.
3. **Micro-Crack Resolving Limit**: Rail surface micro-checks ($<2\text{mm}$) and hairline fastener fatigue cracks cannot be resolved from cab windshield distance.

---

## 5. Recommended Stage 7 AI Architecture

> Dual-Track Hybrid AI Architecture: (1) Supervised Object & Hazard Detection (YOLO) for high-visibility foreign obstacles and near-field missing components; (2) Classical CV & Unsupervised Anomaly Detection (ExG vegetation segmentation, dual-rail polynomial tracking, sleeper periodic variance, ballast GLCM entropy); (3) Multimodal Sensor Fusion to bridge optical limits with replayed IMU vibration and simulated LiDAR/NDE signals.

### Recommended Module Structure for Stage 7:
1. `ai_pipeline/detection/hazard_detector.py`: Supervised YOLO-based detector for foreign track obstructions and prominent hazards.
2. `ai_pipeline/detection/vegetation_detector.py`: Color-space & texture segmentation for vegetation encroachment.
3. `ai_pipeline/detection/rail_tracker.py`: Dual-rail polynomial tracking and macroscopic alignment estimation.
4. `ai_pipeline/anomaly/patch_anomaly.py`: Statistical texture & patch-based anomaly scoring for sleeper/ballast irregularities.
5. `ai_pipeline/fusion/telemetry_fusion.py`: Multimodal fusion layer integrating visual defect scores with replayed IMU vibration and simulated NDE/LiDAR feeds.

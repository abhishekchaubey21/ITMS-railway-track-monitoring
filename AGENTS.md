# ITMS — Agent Instructions

## Project
ITMS (Indigenous Contactless Integrated Track Monitoring System)
is a software prototype for AI-assisted railway track inspection.

## Goal
Build a realistic software-only prototype that demonstrates:

Video/Data Input
→ Computer Vision
→ Defect/Anomaly Detection
→ Location Estimation
→ Sensor Data Replay
→ Multimodal Fusion
→ Severity Assessment
→ Track Health
→ Alerts
→ Dashboard

## Development Principles

1. Build incrementally.
2. Keep components modular.
3. Do not introduce unnecessary dependencies.
4. Prefer simple, explainable solutions over unnecessarily complex ones.
5. Never claim simulated/replayed sensor data is real sensor data.
6. Clearly distinguish:
   - real recorded video
   - public datasets
   - simulated data
   - replayed sensor data
7. Do not fabricate experimental results.
8. Record dataset sources and licenses.
9. Never commit API keys, passwords, tokens, or credentials.
10. Do not download large datasets without approval.
11. Do not modify trained models or model architecture without documenting the reason.
12. Keep AI inference separate from the frontend.
13. Every major feature should have a test or reproducible validation procedure.
14. Preserve existing working functionality when adding new features.

## Architecture

Frontend:
React + Vite

Backend:
Python + FastAPI

AI:
PyTorch + Ultralytics + OpenCV

Database:
SQLite initially, PostgreSQL later if required.

Training:
Local NVIDIA GPU when practical.
Google Colab for GPU-heavy training.

## AI Strategy

Use two complementary approaches:

1. Supervised detection for known defect classes.
2. Anomaly detection for previously unseen abnormalities.

Never claim that the system detects every possible railway defect.

## Data Strategy

Primary demonstration:
Indian railway front-facing train POV video.

Additional datasets may be used for:
- railway defects
- LiDAR/geometry measurements
- IMU/vibration data
- GPS/location data

Different datasets must not be presented as synchronized measurements of the same physical event unless synchronization is actually established.

## Git

Use meaningful commits.

Examples:

feat: add video frame extraction
feat: add defect inference pipeline
fix: handle corrupted video frames
docs: update architecture

Do not make huge unrelated commits.

## Before completing a task

1. Check the existing code.
2. Make the smallest appropriate change.
3. Run relevant tests.
4. Verify the application still works.
5. Explain what changed.
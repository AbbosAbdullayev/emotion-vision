![Demo](assets/face_sample.gif)

## Highlights

- **Real-time inference pipeline** — dedicated `QThread` decouples camera capture and CNN inference from the UI event loop, maintaining smooth frame rendering during continuous prediction
- **Face detection & landmark tracking** — MediaPipe-based detection feeding a 48×48 normalized ROI into a Keras CNN trained on 5-class emotion labels (anger, happiness, sadness, surprise, neutral)
- **Confidence-aware visualization** — live vertical confidence bar rendered per detected face, alongside bounding box with corner-accent styling for a modern HUD aesthetic
- **Emotion-gated capture system** — user selects a target emotion; the system automatically triggers a full-resolution, unprocessed frame capture only when the detected emotion matches, with native save-to-disk via OpenCV
- **Modular, config-driven architecture** — separated UI, inference, and configuration layers; no hardcoded paths, fully portable across machines
- **Modern dark-themed UI** — custom-built PyQt5 interface (no Qt Designer dependency) with a structured sidebar, session timer, and live status indicators
## Planned extensions
- SQLite-backed session logging with historical statistics dashboard
- Multi-face detection with per-face emotion labeling
- Head pose estimation (yaw / pitch / roll) via MediaPipe face mesh
- Attention scoring based on gaze direction
- Age & gender estimation overlay via DeepFace
- Background segmentation for privacy-preserving capture
  
## Reference ##
1.https://github.com/KavenLee/wpod_ocr

2.https://github.com/sergiomsilva/alpr-unconstrained

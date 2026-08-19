import cv2
from pathlib import Path

from ai_pipeline.detection.video_reader import VideoReader
from ai_pipeline.detection.vision_pipeline import VisionPipeline

video = "data/raw/train_pov.mp4"
out_dir = Path("data/processed/real_video_check")
out_dir.mkdir(parents=True, exist_ok=True)

reader = VideoReader(video)
metadata = reader.get_metadata()
pipeline = VisionPipeline()

indices = [
    0,
    metadata.total_frames // 4,
    metadata.total_frames // 2,
    (metadata.total_frames * 3) // 4,
    metadata.total_frames - 1,
]

cap = cv2.VideoCapture(video)

for frame_id in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()

    if not ok:
        continue

    result = pipeline.process_frame(
        frame,
        frame_id=frame_id,
        source_frame=frame_id,
        timestamp=frame_id / metadata.fps,
    )

    annotated = frame.copy()

    for detection in result.detections:
        x1, y1, x2, y2 = map(int, detection.bbox)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.putText(
            annotated,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    output = out_dir / f"frame_{frame_id:05d}.jpg"
    cv2.imwrite(str(output), annotated)

    print(f"saved: {output}")

cap.release()
reader.close()

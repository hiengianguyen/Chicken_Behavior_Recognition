from pathlib import Path

from src.tracker.pipeline import TrackingPipeline

# ==========================
# Đường dẫn
# ==========================

MODEL_PATH = "models/best.pt"

VIDEO_DIR = Path("datasets/videos")
OUTPUT_DIR = Path("datasets/trajectory")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Danh sách video
# ==========================

videos = [
    # "normal.mp4",
    # "crowding.mp4",
    # "standing.mp4",
]

# ==========================
# Tracking
# ==========================

for video in videos:

    print("=" * 50)
    print(f"Processing: {video}")

    pipeline = TrackingPipeline(
        model_path=MODEL_PATH, video_path=str(VIDEO_DIR / video)
    )

    output_csv = OUTPUT_DIR / f"{Path(video).stem}.csv"

    pipeline.run(str(output_csv))

    print(f"Saved -> {output_csv}")

print("=" * 50)
print("Tracking Finished!")

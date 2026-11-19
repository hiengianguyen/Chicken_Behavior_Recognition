from src.tracker.pipeline import TrackingPipeline

pipeline = TrackingPipeline(
    model_path="best_obj.pt", video_path="datasets/videos/normal.mp4"
)

pipeline.run()

from src.tracker.pipeline import TrackingPipeline

pipeline = TrackingPipeline(
    model_path="best.pt", video_path="datasets/videos/crowding.mp4"
)

pipeline.run(output_csv="datasets/trajectory/crowding.csv")

print("Tracking completed!")

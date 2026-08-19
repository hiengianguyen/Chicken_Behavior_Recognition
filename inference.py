"""
Real-time Chicken Behavior Inference Pipeline
Combines:
  - YOLO object detection + ByteTrack tracking
  - Behavior feature extraction (speed, acceleration, standing time)
  - LSTM behavior prediction (Normal / Standing)
  - Status tracking and logging
  - Real-time visualization
"""

import cv2
import argparse
from pathlib import Path
import time
from tqdm import tqdm

from src.pipeline.pipeline import ChickenPipeline


class ChickenInferencePipeline:
    """Main inference orchestrator for real-time processing"""

    def __init__(
        self,
        detector_model="models/best.pt",
        behavior_model="weights/best_model.pt",
        output_video=None,
        log_dir="logs",
    ):

        self.pipeline = ChickenPipeline(
            detector_model=detector_model,
            behavior_model=behavior_model,
        )
        self.output_video = output_video
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.video_writer = None
        self.frame_count = 0
        self.fps_history = []

    def process_video(self, video_path, output_path=None):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*50}")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"{'='*50}\n")

        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )
            print(f"Output video: {output_path}\n")

        start_time = time.time()

        with tqdm(total=total_frames, desc="Inference") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Process frame through pipeline
                frame_start = time.time()
                processed_frame = self.pipeline.process(frame)
                frame_time = time.time() - frame_start

                # Calculate FPS
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(current_fps)

                # Draw FPS on frame
                avg_fps = sum(self.fps_history[-30:]) / len(self.fps_history[-30:])
                cv2.putText(
                    processed_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Save frame if output writer exists
                if self.video_writer:
                    self.video_writer.write(processed_frame)

                # Display frame
                cv2.imshow("Chicken Inference", processed_frame)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                self.frame_count += 1
                pbar.update(1)

        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"\n{'='*50}")
        print("Inference Completed")
        print(f"Frames processed: {self.frame_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if output_path:
            print(f"Output saved: {output_path}")
        print(f"{'='*50}\n")

    def process_webcam(self, duration=None):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open webcam")

        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\n{'='*50}")
        print("Live Webcam Inference")
        print(f"FPS: {fps}")
        print("Press 'q' to quit")
        print(f"{'='*50}\n")

        start_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Check duration
            if duration and (time.time() - start_time) > duration:
                break

            # Process frame through pipeline
            frame_start = time.time()
            processed_frame = self.pipeline.process(frame)
            frame_time = time.time() - frame_start

            # Calculate FPS
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(current_fps)

            # Draw FPS on frame
            avg_fps = sum(self.fps_history[-30:]) / len(self.fps_history[-30:])
            cv2.putText(
                processed_frame,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Display frame
            cv2.imshow("Chicken Inference - Webcam", processed_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"\n{'='*50}")
        print("Webcam Inference Stopped")
        print(f"Frames processed: {self.frame_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Real-time Chicken Behavior Inference")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Input source: video path or "0" for webcam (default: 0)',
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/best.pt",
        help="Path to YOLO detector model (default: models/best.pt)",
    )
    parser.add_argument(
        "--behavior-model",
        type=str,
        default="weights/best_model.pt",
        help="Path to LSTM behavior model (default: weights/best_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (optional)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save logs (default: logs)",
    )

    args = parser.parse_args()

    # Create inference pipeline
    inferencer = ChickenInferencePipeline(
        detector_model=args.detector_model,
        behavior_model=args.behavior_model,
        output_video=args.output,
        log_dir=args.log_dir,
    )

    # Determine source: webcam or video file
    if args.source == "0" or args.source.isdigit():
        # Webcam
        inferencer.process_webcam()
    else:
        # Video file
        if not Path(args.source).exists():
            raise FileNotFoundError(f"Video file not found: {args.source}")

        # Auto-generate output filename if not provided
        output_path = args.output
        if output_path is None and args.output != "none":
            input_path = Path(args.source)
            output_path = f"output_{input_path.stem}.mp4"

        inferencer.process_video(args.source, output_path)


if __name__ == "__main__":
    main()

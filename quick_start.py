import subprocess
import sys
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def setup_environment():
    """Setup and activate virtual environment"""
    print_header("1. Setup Environment")

    print("Creating/activating virtual environment...")
    print("\nWindows:")
    print("  python -m venv .venv")
    print("  .venv\\Scripts\\activate")

    print("\nLinux/Mac:")
    print("  python -m venv .venv")
    print("  source .venv/bin/activate")

    print("\nInstall dependencies:")
    print("  pip install -r requirements.txt")


def test_pipeline():
    """Test the pipeline"""
    print_header("2. Test Pipeline")

    print("Run tests to verify everything works:")
    print("\n  python test_inference.py")
    print("\nExpected output:")
    print("  ✓ All components present")
    print("  ✓ Context initialized")
    print("  ✓ BehaviorRecord works correctly")
    print("  ✓ Feature extraction works")


def run_inference():
    """Show inference commands"""
    print_header("3. Run Inference")

    print("Video File Processing:")
    print("  python inference.py --source videos/chicken_video.mp4 --output result.mp4")

    print("\nLive Webcam (Press 'q' to quit):")
    print("  python inference.py --source 0")

    print("\nWith Custom Models:")
    print("  python inference.py \\")
    print("    --source 0 \\")
    print("    --detector-model models/best.pt \\")
    print("    --behavior-model weights/best_model.pt")


def train_behavior_model():
    """Show training instructions"""
    print_header("4. Train Behavior Model (Optional)")

    print("Step 1: Extract Features from Trajectory Data")
    print('  python -c "')
    print("from src.sequence.builder import SequenceBuilder")
    print("builder = SequenceBuilder()")
    print("feature_files = [")
    print("    ('datasets/features/normal.csv', 0),")
    print("    ('datasets/features/b_1.csv', 1),")
    print("    # ... add all behavior files")
    print("]")
    print("X, y = builder.build(feature_files)")
    print("builder.save(X, y, 'datasets/sequences')")
    print('"')

    print("\nStep 2: Train LSTM Model")
    print('  python -c "')
    print("from src.training.trainer import Trainer")
    print("trainer = Trainer(")
    print("    dataset_dir='datasets/sequences',")
    print("    batch_size=8,")
    print("    epochs=30,")
    print("    learning_rate=0.001")
    print(")")
    print("trainer.train()")
    print('"')


def show_architecture():
    """Show pipeline architecture"""
    print_header("5. Pipeline Architecture")

    architecture = """
VIDEO/WEBCAM INPUT
        ↓
    [YOLO + ByteTrack]
    Detection & Tracking
        ↓ Track(id, bbox, x, y)
    [BehaviorManager]
    Speed, Acceleration, Standing Time
        ↓ BehaviorRecord
    [BehaviorPredictor - LSTM]
    Normal/Standing Classification
        ↓ BehaviorRecord(prediction, confidence)
    [StatusManager]
    Identity & State Tracking
        ↓ Status Snapshot
    [Visualizer + Logger]
    Draw & Save Results
        ↓
    OUTPUT (Screen/Video/Logs)
    """
    print(architecture)


def show_data_flow():
    """Show detailed data flow"""
    print_header("6. Data Flow Details")

    print("BehaviorRecord Fields:")
    print("  • track_id: Unique chicken ID")
    print("  • x, y: Center position")
    print("  • speed: Movement speed")
    print("  • acceleration: Speed change")
    print("  • standing_time: Time standing still")
    print("  • prediction: 'Normal' or 'Standing' (from LSTM)")
    print("  • confidence: 0.0-1.0 confidence score")

    print("\nLSTM Input:")
    print("  • Window: 90 consecutive frames")
    print("  • Features per frame: 7")
    print("    1. timestamp")
    print("    2. x coordinate")
    print("    3. y coordinate")
    print("    4. speed")
    print("    5. acceleration")
    print("    6. direction")
    print("    7. standing_time")
    print("  • Output: class (0=Normal, 1=Standing)")


def show_troubleshooting():
    """Show common issues and solutions"""
    print_header("7. Troubleshooting")

    issues = [
        ("PyTorch CUDA error", "Make sure PyTorch is installed correctly"),
        (
            "  Solution:",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        ),
        ("", ""),
        ("ModuleNotFoundError: No module named 'ultralytics'", "YOLO not installed"),
        ("  Solution:", "pip install ultralytics"),
        ("", ""),
        ("File not found: weights/best_model", "Model not trained yet"),
        (
            "  Solution:",
            "Train the model or modify inference.py to use weights/best_model.pt",
        ),
        ("", ""),
        ("Slow FPS", "Too many heavy operations"),
        ("  Solution:", "Use smaller YOLO model (e.g., best_1.pt instead of best.pt)"),
        ("", ""),
        ("'q' key not working in video", "Window may lose focus"),
        ("  Solution:", "Click on video window before pressing 'q'"),
    ]

    for issue, solution in issues:
        if issue:
            print(f"  {issue}")
            if solution.startswith("Solution"):
                print(f"  {solution}")


def show_performance():
    """Show performance metrics"""
    print_header("8. Performance Metrics")

    print("Typical Performance (NVIDIA GPU):")
    print("  • YOLO Detection: 30-60ms per frame")
    print("  • Feature Extraction: 1-2ms per track")
    print("  • LSTM Inference: 5-10ms per 90-frame window")
    print("  • Visualization: 2-3ms")
    print("  • Overall FPS: 20-25 FPS")

    print("\nOptimization Tips:")
    print("  1. Run on GPU (set CUDA_VISIBLE_DEVICES=0)")
    print("  2. Use smaller YOLO model for faster detection")
    print("  3. Reduce input resolution")
    print("  4. Batch process videos instead of real-time for offline analysis")


def show_file_structure():
    """Show project file structure"""
    print_header("9. Project Structure")

    structure = """
src/
├── core/
│   └── context.py              # Global state (frame, fps, timestamp)
├── main_tracker/
│   └── tracker.py              # YOLO + ByteTrack wrapper
├── behavior/
│   ├── manager.py              # Feature extraction
│   ├── models.py               # BehaviorRecord dataclass
│   └── features/               # Speed, acceleration, standing
├── predictor/
│   ├── predictor.py            # LSTM inference
│   └── buffer.py               # Sequence buffering
├── pipeline/
│   └── pipeline.py             # Main orchestrator
├── status/
│   ├── manager.py              # Track identities
│   └── logger.py               # Save logs
└── visualization/
    └── visualizer.py           # Draw on frame

inference.py                     # Main entry point
test_inference.py               # Test framework
INFERENCE_GUIDE.md              # Detailed documentation
REFACTORING_SUMMARY.md          # What was changed
quick_start.py                  # This file!
    """
    print(structure)


def main():
    """Main menu"""
    print_header("🐔 Chicken Behavior AI - Quick Start Guide")

    options = {
        "1": ("Setup Environment", setup_environment),
        "2": ("Test Pipeline", test_pipeline),
        "3": ("Run Inference", run_inference),
        "4": ("Train Behavior Model", train_behavior_model),
        "5": ("Show Architecture", show_architecture),
        "6": ("Show Data Flow", show_data_flow),
        "7": ("Troubleshooting", show_troubleshooting),
        "8": ("Performance Metrics", show_performance),
        "9": ("Project Structure", show_file_structure),
        "0": (
            "Show All",
            lambda: [
                f()
                for f in [
                    setup_environment,
                    test_pipeline,
                    run_inference,
                    train_behavior_model,
                    show_architecture,
                    show_data_flow,
                    show_troubleshooting,
                    show_performance,
                    show_file_structure,
                ]
            ],
        ),
        "q": ("Exit", lambda: None),
    }

    while True:
        print("\nMenu:")
        for key, (desc, _) in options.items():
            if key != "q":
                print(f"  [{key}] {desc}")
        print(f"  [q] Exit")

        choice = input("\nSelect option: ").strip().lower()

        if choice == "q":
            print("\nExiting. Happy chicken tracking! 🐔\n")
            break
        elif choice in options:
            _, func = options[choice]
            if func:
                func()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        if sys.argv[1] == "--all":
            setup_environment()
            test_pipeline()
            run_inference()
            train_behavior_model()
            show_architecture()
            show_data_flow()
            show_troubleshooting()
            show_performance()
            show_file_structure()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python quick_start.py [--all]")
    else:
        # Interactive menu
        main()

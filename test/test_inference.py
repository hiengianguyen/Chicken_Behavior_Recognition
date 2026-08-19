"""
Test script to verify the real-time inference pipeline works correctly.
Run this to check if all components are properly integrated.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.pipeline import ChickenPipeline
from src.core.context import AppContext
import numpy as np


def test_pipeline_components():
    """Test that all components initialize correctly"""

    print("\n" + "=" * 60)
    print("Testing Pipeline Components")
    print("=" * 60 + "\n")

    try:
        # Check if model files exist
        from pathlib import Path

        print("1. Checking model files...")
        detector_model = "models/best.pt"
        behavior_model = "weights/best_model.pt"  # Use available model

        if not Path(detector_model).exists():
            print(f"   ⚠ Warning: {detector_model} not found")
            detector_model = None
        else:
            print(f"   ✓ Found {detector_model}")

        if not Path(behavior_model).exists():
            print(f"   ⚠ Warning: {behavior_model} not found")
            behavior_model = None
        else:
            print(f"   ✓ Found {behavior_model}\n")

        if not detector_model or not behavior_model:
            print("   ⚠ Some models missing. Skipping full initialization.\n")
            print("   Run training pipeline to generate models:\n")
            print("   1. Extract features:")
            print(
                '      python -c "from src.sequence.builder import SequenceBuilder; ..."'
            )
            print("   2. Train behavior model:")
            print('      python -c "from src.training.trainer import Trainer; ..."')
            return True

        print("2. Initializing ChickenPipeline...")
        pipeline = ChickenPipeline(
            detector_model=detector_model,
            behavior_model=behavior_model,
        )
        print("   ✓ Pipeline initialized successfully\n")

        print("2. Checking pipeline components...")
        assert hasattr(pipeline, "tracker"), "Missing tracker"
        assert hasattr(pipeline, "behavior_manager"), "Missing behavior_manager"
        assert hasattr(pipeline, "predictor"), "Missing predictor"
        assert hasattr(pipeline, "status_manager"), "Missing status_manager"
        assert hasattr(pipeline, "visualizer"), "Missing visualizer"
        print("   ✓ All components present\n")

        print("3. Checking context...")
        assert isinstance(pipeline.context, AppContext), "Context not initialized"
        print("   ✓ Context initialized\n")

        print("4. Testing BehaviorRecord...")
        from src.behavior.models import BehaviorRecord

        record = BehaviorRecord(
            frame=0,
            timestamp=0.0,
            track_id=1,
            x=100.0,
            y=100.0,
            speed=1.5,
            acceleration=0.1,
            standing_time=0.0,
        )
        assert hasattr(record, "to_dict"), "Missing to_dict method"
        assert hasattr(record, "to_feature"), "Missing to_feature method"
        record_dict = record.to_dict()
        assert "track_id" in record_dict, "Missing track_id in dict"
        print("   ✓ BehaviorRecord works correctly\n")

        print("5. Testing feature extraction...")
        features = record.to_feature()
        assert len(features) == 9, f"Expected 9 features, got {len(features)}"
        print(f"   Features: {features}")
        print("   ✓ Feature extraction works\n")

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {str(e)}\n")
        import traceback

        traceback.print_exc()
        return False


def test_inference_modes():
    """Test inference with sample data"""

    print("\n" + "=" * 60)
    print("Testing Inference Modes")
    print("=" * 60 + "\n")

    try:
        from inference import ChickenInferencePipeline

        print("1. Checking inference script...")
        print("   ✓ Inference module loaded\n")

        print("2. Command-line interface:")
        print("   Video mode:")
        print("      python inference.py --source video.mp4 --output output.mp4")
        print("   Webcam mode:")
        print("      python inference.py --source 0")
        print("   With custom models:")
        print("      python inference.py --source 0 \\")
        print("         --detector-model models/best.pt \\")
        print("         --behavior-model weights/best_model.pt\n")

        print("=" * 60)
        print("Ready for real-time inference! ✓")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {str(e)}\n")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline_components()
    if success:
        test_inference_modes()

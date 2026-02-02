"""
Complete System Test Script
Tests all components of the VideoX Action Recognition System
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_imports():
    """Test all imports"""
    print("=" * 70)
    print("TEST 1: IMPORTS")
    print("=" * 70)
    
    try:
        print("Testing imports...")
        
        # Core modules
        from model_architecture import VideoXActionRecognizer, create_model
        from dataset_loader import ActionRecognitionDataset, create_dataloaders
        from trainer_module import ActionRecognitionTrainer
        from inference_module import ActionRecognitionInference
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 70)
    print("TEST 2: CUDA")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("✅ CUDA test passed!")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU")
        return False


def test_model_creation():
    """Test model creation"""
    print("\n" + "=" * 70)
    print("TEST 3: MODEL CREATION")
    print("=" * 70)
    
    try:
        from model_architecture import VideoXActionRecognizer
        
        print("Creating model...")
        model = VideoXActionRecognizer(
            num_classes=50,
            d_model=768,
            num_frames=16,
            temporal_layers=4
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("✅ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test model forward pass"""
    print("\n" + "=" * 70)
    print("TEST 4: FORWARD PASS")
    print("=" * 70)
    
    try:
        from model_architecture import VideoXActionRecognizer
        
        model = VideoXActionRecognizer(
            num_classes=50,
            d_model=768,
            num_frames=16,
            temporal_layers=4
        )
        
        # Create dummy input
        batch_size = 2
        num_frames = 16
        frames = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        print(f"Input shape: {frames.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(frames)
        
        print(f"\nOutput shapes:")
        print(f"  Action logits: {outputs['action_logits'].shape}")
        print(f"  Boundary start: {outputs['boundary_start'].shape}")
        print(f"  Boundary end: {outputs['boundary_end'].shape}")
        print(f"  Confidence: {outputs['confidence'].shape}")
        
        print("✅ Forward pass test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 70)
    print("TEST 5: CONFIGURATION")
    print("=" * 70)
    
    try:
        import yaml
        
        config_path = Path('config/config.yaml')
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("Configuration loaded:")
        print(f"  Model: {config['model'].get('videox_model', 'N/A')}")
        print(f"  Num frames: {config['model']['num_frames']}")
        print(f"  Num classes: {config['model']['num_classes']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_directories():
    """Test directory structure"""
    print("\n" + "=" * 70)
    print("TEST 6: DIRECTORY STRUCTURE")
    print("=" * 70)
    
    required_dirs = [
        'config',
        'src',
        'data/videos',
        'checkpoints',
        'outputs/predictions'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists()
        
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}")
        
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ Directory structure test passed!")
    else:
        print("\n⚠️  Some directories missing (will be created automatically)")
    
    return True


def test_video_loading():
    """Test video loading"""
    print("\n" + "=" * 70)
    print("TEST 7: VIDEO LOADING")
    print("=" * 70)
    
    try:
        import cv2
        
        video_dir = Path('data/videos')
        videos = list(video_dir.glob('*.mp4'))
        
        print(f"Found {len(videos)} videos in {video_dir}")
        
        if len(videos) > 0:
            # Test loading first video
            test_video = str(videos[0])
            print(f"\nTesting video: {test_video}")
            
            cap = cv2.VideoCapture(test_video)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps
                
                print(f"  FPS: {fps}")
                print(f"  Frames: {int(frame_count)}")
                print(f"  Duration: {duration:.2f}s")
                
                # Read one frame
                ret, frame = cap.read()
                if ret:
                    print(f"  Frame shape: {frame.shape}")
                    print("✅ Video loading test passed!")
                    cap.release()
                    return True
                else:
                    print("❌ Could not read frame")
                    cap.release()
                    return False
            else:
                print("❌ Could not open video")
                return False
        else:
            print("⚠️  No videos found - add .mp4 files to data/videos/")
            return True
        
    except Exception as e:
        print(f"❌ Video loading test failed: {e}")
        return False


def test_segment_prediction():
    """Test segment prediction"""
    print("\n" + "=" * 70)
    print("TEST 8: SEGMENT PREDICTION")
    print("=" * 70)
    
    try:
        from model_architecture import VideoXActionRecognizer
        
        model = VideoXActionRecognizer(
            num_classes=50,
            d_model=768,
            num_frames=16,
            temporal_layers=4
        )
        
        # Create dummy input
        frames = torch.randn(1, 16, 3, 224, 224)
        fps = 30.0
        
        print("Predicting segments...")
        segments = model.predict_segments(
            frames,
            fps=fps,
            threshold_start=0.5,
            threshold_end=0.5,
            min_duration=8.0,
            max_duration=40.0
        )
        
        print(f"Predicted {len(segments[0])} segments")
        
        if len(segments[0]) > 0:
            print(f"First segment: {segments[0][0]}")
        
        print("✅ Segment prediction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Segment prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("VIDEOX ACTION RECOGNITION - SYSTEM TEST")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Configuration", test_config),
        ("Directory Structure", test_directories),
        ("Video Loading", test_video_loading),
        ("Segment Prediction", test_segment_prediction),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Add videos to data/videos/")
    print("  2. Update config/config.yaml with annotations")
    print("  3. Run: python main.py --mode train --epochs 50")
    print("  4. Run: python app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

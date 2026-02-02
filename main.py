"""
Main Entry Point for VideoX Action Recognition System
Complete implementation for Atlas Action Recognition
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_architecture import create_model
from dataset_loader import create_dataloaders
from trainer_module import ActionRecognitionTrainer
from inference_module import ActionRecognitionInference


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Prepare dataset"""
    print("=" * 70)
    print("STAGE 1: DATA PREPARATION")
    print("=" * 70)
    
    # Create directories
    Path(config['data']['video_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['annotations_dir']).mkdir(parents=True, exist_ok=True)
    
    # Count videos
    video_dir = Path(config['data']['video_dir'])
    videos = list(video_dir.glob('*.mp4'))
    
    print(f"Found {len(videos)} videos in {video_dir}")
    print(f"Annotations for {len(config['annotations_raw'])} videos")
    
    if len(videos) == 0:
        print("\n⚠️  WARNING: No videos found!")
        print(f"   Please add .mp4 files to: {video_dir}")
        return False
    
    print("\n✅ Data preparation complete!")
    return True


def train_model(config: dict, resume: bool = False):
    """Train the model"""
    print("=" * 70)
    print("STAGE 2: TRAINING")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config)
    
    # Create trainer
    trainer = ActionRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume if specified
    resume_path = None
    if resume:
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        latest_checkpoint = checkpoint_dir / 'latest.pth'
        
        if latest_checkpoint.exists():
            resume_path = str(latest_checkpoint)
            print(f"\nResuming from: {resume_path}")
    
    # Train
    trainer.train(resume_from=resume_path)
    
    print("\n✅ Training complete!")


def predict(config: dict, video_path: str = None):
    """Make predictions"""
    print("=" * 70)
    print("STAGE 3: PREDICTION")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint_path = Path(config['training']['checkpoint_dir']) / 'best.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("   Please train the model first: python main.py --mode train")
        return
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Create inference engine
    inference = ActionRecognitionInference(
        model=model,
        config=config,
        device=device,
        checkpoint_path=str(checkpoint_path)
    )
    
    # Get videos to process
    if video_path:
        video_paths = [video_path]
    else:
        video_dir = Path(config['data']['video_dir'])
        video_paths = list(video_dir.glob('*.mp4'))
    
    if not video_paths:
        print("❌ No videos found to process")
        return
    
    print(f"\nProcessing {len(video_paths)} video(s)...")
    
    # Process videos
    for vp in video_paths:
        print(f"\nProcessing: {vp}")
        result = inference.predict_video(str(vp))
        
        # Save predictions
        inference.save_predictions(result)
        
        # Print summary
        print(f"✅ Found {result['num_segments']} segments")
        print(f"   Duration: {result['duration']}s")
        
        # Print first 3 segments
        for seg in result['segments'][:3]:
            print(f"   {seg['start']}-{seg['end']}s: {seg['action']} ({seg['confidence']:.2%})")
    
    print("\n✅ Prediction complete!")


def evaluate(config: dict):
    """Evaluate predictions"""
    print("=" * 70)
    print("STAGE 4: EVALUATION")
    print("=" * 70)
    
    print("Evaluating predictions...")
    
    # Simple evaluation (you can extend this)
    predictions_dir = Path('outputs/predictions')
    
    if not predictions_dir.exists():
        print("❌ No predictions found")
        print("   Please run predictions first: python main.py --mode predict")
        return
    
    predictions = list(predictions_dir.glob('*.json'))
    print(f"Found {len(predictions)} prediction files")
    
    # Calculate basic stats
    total_segments = 0
    total_confidence = 0
    
    import json
    for pred_file in predictions:
        with open(pred_file, 'r') as f:
            pred = json.load(f)
            total_segments += pred['num_segments']
            
            for seg in pred['segments']:
                total_confidence += seg['confidence']
    
    if total_segments > 0:
        avg_confidence = total_confidence / total_segments
        
        results = {
            'total_videos': len(predictions),
            'total_segments': total_segments,
            'avg_segments_per_video': total_segments / len(predictions),
            'avg_confidence': round(avg_confidence, 3)
        }
        
        print(f"\nResults: {results}")
        
        # Save results
        output_dir = Path('outputs/evaluations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Evaluation complete!")
    else:
        print("⚠️  No segments found in predictions")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VideoX Action Recognition System')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['prepare', 'train', 'predict', 'evaluate', 'all'],
                       help='Operation mode')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    
    parser.add_argument('--video', type=str, default=None,
                       help='Specific video to predict')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Execute based on mode
    if args.mode == 'prepare' or args.mode == 'all':
        if not prepare_data(config):
            return
    
    if args.mode == 'train' or args.mode == 'all':
        train_model(config, resume=args.resume)
    
    if args.mode == 'predict' or args.mode == 'all':
        predict(config, video_path=args.video)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        evaluate(config)
    
    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

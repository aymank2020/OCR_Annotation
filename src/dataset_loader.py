"""
Video Dataset Loader with Real Video Processing
Complete implementation for Atlas Action Recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
import re


class ActionRecognitionDataset(Dataset):
    """
    Dataset for loading videos with action annotations
    
    Features:
    - Real video frame extraction (OpenCV)
    - Temporal segment parsing from Atlas format
    - Automatic train/val splitting
    - Frame sampling strategies
    """
    
    def __init__(self,
                 video_dir: str,
                 annotations: Dict[str, str],
                 num_frames: int = 16,
                 frame_size: Tuple[int, int] = (224, 224),
                 split: str = 'train',
                 augment: bool = False):
        
        self.video_dir = Path(video_dir)
        self.annotations = annotations
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split
        self.augment = augment
        
        # Parse all annotations
        self.samples = self._parse_annotations()
        
        print(f"{split.upper()} Dataset: {len(self.samples)} segments from {len(annotations)} videos")
    
    def _parse_annotations(self) -> List[Dict]:
        """Parse annotations from Atlas format"""
        samples = []
        
        for video_id, annotation_text in self.annotations.items():
            video_path = self.video_dir / f"{video_id}.mp4"
            
            if not video_path.exists():
                print(f"⚠️  Warning: Video not found: {video_path}")
                continue
            
            # Parse segments
            segments = self._parse_segments(annotation_text)
            
            for segment in segments:
                samples.append({
                    'video_path': str(video_path),
                    'video_id': video_id,
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'action_text': segment['action'],
                    'action_id': segment['action_id']
                })
        
        return samples
    
    def _parse_segments(self, annotation_text: str) -> List[Dict]:
        """
        Parse Atlas format annotations
        Format: 0:00.0-0:20.0#1 Action description
        """
        segments = []
        
        for line in annotation_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Parse time range and action
            match = re.match(r'(\d+):(\d+\.\d+)-(\d+):(\d+\.\d+)#(\d+)\s+(.+)', line)
            if match:
                start_min, start_sec, end_min, end_sec, action_id, action_text = match.groups()
                
                start_time = float(start_min) * 60 + float(start_sec)
                end_time = float(end_min) * 60 + float(end_sec)
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'action_id': int(action_id) - 1,  # 0-indexed
                    'action': action_text.strip()
                })
        
        return segments
    
    def _extract_frames(self, video_path: str, start_time: float, end_time: float) -> np.ndarray:
        """
        Extract frames from video segment
        
        Returns:
            frames: [T, H, W, C] numpy array
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame
        
        # Sample frame indices uniformly
        if total_frames <= self.num_frames:
            # Repeat frames if too few
            frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        else:
            # Sample uniformly
            frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize
                frame = cv2.resize(frame, self.frame_size)
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Use black frame if failed
                frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        cap.release()
        
        return np.stack(frames, axis=0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single sample
        
        Returns:
            frames: [T, C, H, W] tensor
            action_id: int
            metadata: dict
        """
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(
            sample['video_path'],
            sample['start_time'],
            sample['end_time']
        )  # [T, H, W, C]
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Apply augmentation if training
        if self.augment and self.split == 'train':
            frames = self._augment_frames(frames)
        
        # Convert to tensor and permute to [T, C, H, W]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        metadata = {
            'video_id': sample['video_id'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'action_text': sample['action_text']
        }
        
        return frames, sample['action_id'], metadata
    
    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """Simple data augmentation"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        # Random brightness
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 1)
        
        return frames


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader
    """
    # Load annotations
    annotations = config['annotations_raw']
    
    # Split into train/val (80/20)
    video_ids = list(annotations.keys())
    np.random.seed(42)
    np.random.shuffle(video_ids)
    
    split_idx = int(len(video_ids) * 0.8)
    train_ids = video_ids[:split_idx]
    val_ids = video_ids[split_idx:]
    
    train_annotations = {vid: annotations[vid] for vid in train_ids}
    val_annotations = {vid: annotations[vid] for vid in val_ids}
    
    # Create datasets
    train_dataset = ActionRecognitionDataset(
        video_dir=config['data']['video_dir'],
        annotations=train_annotations,
        num_frames=config['model']['num_frames'],
        frame_size=tuple(config['model']['frame_size']),
        split='train',
        augment=True
    )
    
    val_dataset = ActionRecognitionDataset(
        video_dir=config['data']['video_dir'],
        annotations=val_annotations,
        num_frames=config['model']['num_frames'],
        frame_size=tuple(config['model']['frame_size']),
        split='val',
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset loader...")
    
    config = yaml.safe_load(open('config/config.yaml', 'r'))
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch
    frames, actions, metadata = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Sample action: {metadata['action_text'][0]}")
    
    print("\n✅ Dataset test passed!")

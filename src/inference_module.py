"""
Inference Module with VideoX Support
Complete implementation for Atlas Action Recognition
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


class ActionRecognitionInference:
    """
    Inference engine for action recognition
    
    Features:
    - Video segment prediction
    - Temporal boundary detection
    - Atlas format output
    - Dense captioning (VideoX only)
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda',
                 checkpoint_path: Optional[str] = None):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Inference config
        self.num_frames = config['model']['num_frames']
        self.frame_size = tuple(config['model']['frame_size'])
        
        self.confidence_threshold = config['inference']['confidence_threshold']
        self.boundary_start_threshold = config['inference']['boundary_start_threshold']
        self.boundary_end_threshold = config['inference']['boundary_end_threshold']
        self.min_duration = config['inference']['min_action_duration']
        self.max_duration = config['inference']['max_action_duration']
        self.generate_captions = config['inference'].get('generate_captions', False)
        
        # Action vocabulary (if available)
        self.action_vocab = self._load_vocabulary()
        
        print("Inference engine initialized")
        print(f"  Device: {device}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Min duration: {self.min_duration}s")
        print(f"  Max duration: {self.max_duration}s")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from {checkpoint_path}")
            print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('best_loss', 'unknown'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from {checkpoint_path}")
    
    def _load_vocabulary(self) -> Dict[int, str]:
        """Load action vocabulary"""
        vocab_path = Path('data/vocabulary.yaml')
        
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab = yaml.safe_load(f)
                return vocab.get('id_to_action', {})
        
        return {}
    
    def extract_frames(self, video_path: str, max_duration: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Extract frames from entire video or limited duration
        
        Returns:
            frames: [T, H, W, C] numpy array
            fps: video FPS
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit duration if specified
        if max_duration:
            max_frames = int(max_duration * fps)
            total_frames = min(total_frames, max_frames)
        
        # Sample frame indices
        if total_frames <= self.num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        cap.release()
        
        return np.stack(frames, axis=0), fps
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess frames for model input
        
        Args:
            frames: [T, H, W, C] numpy array
            
        Returns:
            tensor: [1, T, C, H, W]
        """
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        
        # To tensor [T, C, H, W]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        # Add batch dimension
        frames = frames.unsqueeze(0)  # [1, T, C, H, W]
        
        return frames
    
    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict:
        """
        Predict actions for entire video
        
        Returns:
            Dictionary with predictions in multiple formats
        """
        # Extract frames
        frames_np, fps = self.extract_frames(video_path)
        
        # Preprocess
        frames = self.preprocess_frames(frames_np).to(self.device)
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        
        # Predict
        outputs = self.model(frames, generate_captions=self.generate_captions)
        
        # Parse predictions
        segments = self._parse_predictions(outputs, fps, duration)
        
        # Format results
        result = {
            'video_path': video_path,
            'video_id': Path(video_path).stem,
            'duration': round(duration, 1),
            'fps': fps,
            'num_segments': len(segments),
            'segments': segments
        }
        
        # Add formatted strings
        result['atlas_format'] = self._to_atlas_format(segments)
        result['json_format'] = segments
        
        return result
    
    def _parse_predictions(self, outputs: Dict, fps: float, duration: float) -> List[Dict]:
        """Parse model outputs into segments"""
        # Get predictions
        action_logits = outputs['action_logits'].squeeze(0)  # [T, num_classes]
        boundary_start = outputs['boundary_start'].squeeze()  # [T]
        boundary_end = outputs['boundary_end'].squeeze()      # [T]
        confidence = outputs['confidence'].squeeze()          # [T]
        
        # Get action predictions
        action_probs = torch.softmax(action_logits, dim=-1)
        action_ids = torch.argmax(action_probs, dim=-1).cpu().numpy()
        
        boundary_start = boundary_start.cpu().numpy()
        boundary_end = boundary_end.cpu().numpy()
        confidence = confidence.cpu().numpy()
        
        # Find segments
        segments = []
        num_frames = len(action_ids)
        in_segment = False
        segment_start = 0
        current_action = None
        
        for t in range(num_frames):
            is_start = boundary_start[t] > self.boundary_start_threshold
            is_end = boundary_end[t] > self.boundary_end_threshold
            
            # Start new segment
            if is_start and not in_segment:
                in_segment = True
                segment_start = t
                current_action = action_ids[t]
            
            # End segment
            elif is_end and in_segment:
                segment_end = t
                
                # Calculate times
                start_time = (segment_start / num_frames) * duration
                end_time = (segment_end / num_frames) * duration
                seg_duration = end_time - start_time
                
                # Check duration constraints
                if self.min_duration <= seg_duration <= self.max_duration:
                    avg_conf = confidence[segment_start:segment_end].mean()
                    
                    # Check confidence
                    if avg_conf >= self.confidence_threshold:
                        # Get action text
                        action_text = self.action_vocab.get(
                            int(current_action),
                            f"Action_{current_action}"
                        )
                        
                        segment = {
                            'start': round(start_time, 1),
                            'end': round(end_time, 1),
                            'duration': round(seg_duration, 1),
                            'action': action_text,
                            'action_id': int(current_action),
                            'confidence': round(float(avg_conf), 3)
                        }
                        
                        segments.append(segment)
                
                in_segment = False
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        return segments
    
    def _to_atlas_format(self, segments: List[Dict]) -> str:
        """
        Convert segments to Atlas format
        
        Format: 0:00.0-0:20.0#1 Action description
        """
        lines = []
        
        for i, seg in enumerate(segments, 1):
            start_min = int(seg['start'] // 60)
            start_sec = seg['start'] % 60
            end_min = int(seg['end'] // 60)
            end_sec = seg['end'] % 60
            
            line = f"{start_min}:{start_sec:04.1f}-{end_min}:{end_sec:04.1f}#{i} {seg['action']}"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict]:
        """Predict for multiple videos"""
        results = []
        
        for video_path in video_paths:
            print(f"Processing: {video_path}")
            result = self.predict_video(video_path)
            results.append(result)
        
        return results
    
    def save_predictions(self, predictions: Dict, output_dir: str = 'outputs/predictions'):
        """Save predictions to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_id = predictions['video_id']
        
        # Save JSON
        import json
        with open(output_dir / f"{video_id}.json", 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save Atlas format
        with open(output_dir / f"{video_id}_atlas.txt", 'w') as f:
            f.write(predictions['atlas_format'])
        
        print(f"✅ Saved predictions for {video_id}")


if __name__ == "__main__":
    print("Inference module ready!")
    print("Import this module to make predictions.")

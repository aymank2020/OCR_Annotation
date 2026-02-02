"""
VideoX-based Action Recognition Model
Complete implementation for Atlas Action Recognition
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Try to import VideoX
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'VideoX'))
    from videox.models import VideoXModel
    from videox.processors import VideoXProcessor
    VIDEOX_AVAILABLE = True
except ImportError:
    print("⚠️  VideoX not found, falling back to CLIP")
    from transformers import CLIPModel, CLIPProcessor
    VIDEOX_AVAILABLE = False


class VideoXActionRecognizer(nn.Module):
    """
    Action Recognition using Microsoft VideoX or CLIP fallback
    
    Features:
    - Temporal action localization
    - Dense video captioning (VideoX only)
    - Multi-modal understanding
    - Automatic fallback to CLIP if VideoX unavailable
    """
    
    def __init__(self,
                 model_name="microsoft/videox-base",
                 clip_model_name="openai/clip-vit-base-patch32",
                 num_classes=50,
                 d_model=768,
                 num_frames=16,
                 temporal_layers=4,
                 dropout=0.1,
                 max_duration=40):
        super().__init__()
        
        self.num_frames = num_frames
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_duration = max_duration
        
        # Initialize model (VideoX or CLIP)
        if VIDEOX_AVAILABLE:
            print(f"Loading VideoX model: {model_name}...")
            self._init_videox(model_name)
            self.model_type = 'videox'
        else:
            print(f"Loading CLIP model: {clip_model_name}...")
            self._init_clip(clip_model_name)
            self.model_type = 'clip'
        
        # Get feature dimension
        feature_dim = 768 if self.model_type == 'videox' else 512
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers
        )
        
        # Action classifier
        self.action_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Boundary detector (start/end probabilities)
        self.boundary_detector = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, 2)  # بدون Sigmoid
)
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Print model info
        print(f"Model initialized:")
        print(f"  - Model type: {self.model_type.upper()}")
        print(f"  - Feature dimension: {feature_dim}D")
        print(f"  - Model dimension: {d_model}D")
        print(f"  - Temporal layers: {temporal_layers}")
        print(f"  - Number of classes: {num_classes}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def _init_videox(self, model_name: str):
        """Initialize VideoX model"""
        try:
            self.backbone = VideoXModel.from_pretrained(model_name)
            self.processor = VideoXProcessor.from_pretrained(model_name)
            
            # Freeze backbone initially
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            self.use_captioning = True
            print("  ✅ VideoX loaded with captioning support")
            
        except Exception as e:
            print(f"  ⚠️  Failed to load VideoX: {e}")
            print("  Falling back to CLIP...")
            self._init_clip("openai/clip-vit-base-patch32")
            self.model_type = 'clip'
    
    def _init_clip(self, model_name: str):
        """Initialize CLIP model (fallback)"""
        self.backbone = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.use_captioning = False
        print("  ✅ CLIP loaded (no captioning)")
    
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from video frames
        
        Args:
            frames: [B, T, C, H, W] - video frames
            
        Returns:
            features: [B, T, D] - temporal features
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        if self.model_type == 'videox':
            # VideoX processes full video
            with torch.no_grad():
                outputs = self.backbone.get_video_features(
                    pixel_values=frames,
                    return_dict=True
                )
                features = outputs.last_hidden_state  # [B, T, D]
        else:
            # CLIP processes frame by frame
            frames_flat = frames.view(batch_size * num_frames, C, H, W)
            
            with torch.no_grad():
                vision_outputs = self.backbone.vision_model(pixel_values=frames_flat)
                clip_features = self.backbone.visual_projection(vision_outputs.pooler_output)
            
            # Reshape to temporal
            features = clip_features.view(batch_size, num_frames, -1)
        
        return features
    
    def forward(self, 
                frames: torch.Tensor,
                generate_captions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            frames: [B, T, C, H, W] - video frames
            generate_captions: Generate text descriptions (VideoX only)
            
        Returns:
            dict with predictions
        """
        batch_size = frames.size(0)
        
        # Extract features
        raw_features = self.extract_features(frames)  # [B, T, D]
        
        # Project features
        features = self.feature_projection(raw_features)  # [B, T, d_model]
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(features)  # [B, T, d_model]
        
        # Action classification
        action_logits = self.action_classifier(temporal_features)  # [B, T, num_classes]
        
        # Boundary detection
        boundaries = self.boundary_detector(temporal_features)  # [B, T, 2]
        boundary_start = boundaries[..., 0:1]  # [B, T, 1]
        boundary_end = boundaries[..., 1:2]    # [B, T, 1]
        
        # Confidence estimation
        confidence = self.confidence_head(temporal_features)  # [B, T, 1]
        
        output = {
            'action_logits': action_logits,
            'boundary_start': boundary_start,
            'boundary_end': boundary_end,
            'confidence': confidence,
            'features': temporal_features
        }
        
        # Optional captions (VideoX only)
        if generate_captions and self.use_captioning and self.model_type == 'videox':
            try:
                captions = self.generate_captions(frames)
                output['captions'] = captions
            except:
                pass
        
        return output
    
    def generate_captions(self, frames: torch.Tensor, max_length: int = 30) -> List[str]:
        """
        Generate action descriptions (VideoX only)
        
        Args:
            frames: [B, T, C, H, W]
            max_length: Max caption length
            
        Returns:
            List of captions
        """
        if self.model_type != 'videox' or not self.use_captioning:
            return []
        
        with torch.no_grad():
            outputs = self.backbone.generate(
                pixel_values=frames,
                max_length=max_length,
                num_beams=3,
                early_stopping=True
            )
            
            captions = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )
        
        return captions
    
    def predict_segments(self,
                        frames: torch.Tensor,
                        fps: float,
                        threshold_start: float = 0.5,
                        threshold_end: float = 0.5,
                        min_duration: float = 8.0,
                        max_duration: float = 40.0) -> List[List[Dict]]:
        """
        Predict action segments with boundaries
        
        Args:
            frames: [B, T, C, H, W]
            fps: Video FPS
            threshold_start: Start detection threshold
            threshold_end: End detection threshold
            min_duration: Min segment duration (seconds)
            max_duration: Max segment duration (seconds)
            
        Returns:
            List of segments per batch
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(frames, generate_captions=self.use_captioning)
            
            action_probs = torch.softmax(outputs['action_logits'], dim=-1)
            action_preds = torch.argmax(action_probs, dim=-1).cpu().numpy()
            
            boundary_start = outputs['boundary_start'].squeeze(-1).cpu().numpy()
            boundary_end = outputs['boundary_end'].squeeze(-1).cpu().numpy()
            confidence = outputs['confidence'].squeeze(-1).cpu().numpy()
            
            captions = outputs.get('captions', None)
        
        batch_size, num_frames = action_preds.shape
        all_segments = []
        
        for b in range(batch_size):
            segments = []
            in_segment = False
            segment_start = 0
            current_action = None
            
            for t in range(num_frames):
                is_start = boundary_start[b, t] > threshold_start
                is_end = boundary_end[b, t] > threshold_end
                
                if is_start and not in_segment:
                    in_segment = True
                    segment_start = t
                    current_action = action_preds[b, t]
                
                elif is_end and in_segment:
                    segment_end = t
                    
                    # Calculate times
                    start_time = (segment_start / num_frames) * (num_frames / fps)
                    end_time = (segment_end / num_frames) * (num_frames / fps)
                    duration = end_time - start_time
                    
                    if min_duration <= duration <= max_duration:
                        avg_confidence = confidence[b, segment_start:segment_end].mean()
                        
                        segment = {
                            'start': round(start_time, 1),
                            'end': round(end_time, 1),
                            'duration': round(duration, 1),
                            'action_id': int(current_action),
                            'confidence': round(float(avg_confidence), 3)
                        }
                        
                        if captions and b < len(captions):
                            segment['description'] = captions[b]
                        
                        segments.append(segment)
                    
                    in_segment = False
            
            all_segments.append(segments)
        
        return all_segments
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✅ Backbone unfrozen for fine-tuning")


def create_model(config):
    """Create model from config"""
    model = VideoXActionRecognizer(
        model_name=config['model'].get('videox_model', 'microsoft/videox-base'),
        clip_model_name=config['model'].get('clip_model', 'openai/clip-vit-base-patch32'),
        num_classes=config['model'].get('num_classes', 50),
        d_model=config['model']['d_model'],
        num_frames=config['model']['num_frames'],
        temporal_layers=config['model']['temporal_layers'],
        dropout=config['model'].get('dropout', 0.1),
        max_duration=config['easy_mode']['max_duration']
    )
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing VideoX Action Recognizer...")
    
    model = VideoXActionRecognizer(
        num_classes=50,
        d_model=768,
        num_frames=16,
        temporal_layers=4
    )
    
    # Test forward pass
    frames = torch.randn(2, 16, 3, 224, 224)
    outputs = model(frames)
    
    print(f"\nOutput shapes:")
    print(f"  Action logits: {outputs['action_logits'].shape}")
    print(f"  Boundary start: {outputs['boundary_start'].shape}")
    print(f"  Boundary end: {outputs['boundary_end'].shape}")
    print(f"  Confidence: {outputs['confidence'].shape}")
    
    # Test segment prediction
    segments = model.predict_segments(frames, fps=30.0)
    print(f"\nPredicted segments: {len(segments[0])}")
    
    print("\n✅ Model test passed!")

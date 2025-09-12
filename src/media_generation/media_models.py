"""Data models for media generation pipeline"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class AudioQuality(str, Enum):
    """Audio quality settings"""
    HIGH = "high"
    MEDIUM = "medium"
    FAST = "fast"


class ImageStyle(str, Enum):
    """Image generation styles"""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    DOCUMENTARY = "documentary"
    CINEMATIC = "cinematic"


class AudioSegment(BaseModel):
    """Individual audio segment"""
    id: str
    text: str
    file_path: Optional[str] = None
    start_time: float = 0.0  # seconds
    duration: float = 0.0  # seconds
    speaker_voice: str = "default"
    processing_settings: Dict[str, Any] = {}
    
    
class GeneratedImage(BaseModel):
    """Generated image with metadata"""
    id: str
    prompt: str
    file_path: Optional[str] = None
    timestamp: float  # When it should appear in video
    duration: float = 8.0  # How long to show
    style_used: str = ""
    generation_settings: Dict[str, Any] = {}
    quality_score: float = 0.0
    

class AudioGenerationRequest(BaseModel):
    """Request for audio generation"""
    script_text: str
    voice_model: str = "default"
    quality: AudioQuality = AudioQuality.HIGH
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 0.8
    output_format: str = "wav"
    sample_rate: int = 44100
    

class ImageGenerationRequest(BaseModel):
    """Request for image generation"""
    prompts: List[str]
    style: str = "documentary"
    resolution: str = "1920x1080"
    batch_size: int = 4
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    negative_prompt: str = ""
    seed: Optional[int] = None
    

class MediaGenerationResult(BaseModel):
    """Complete media generation result"""
    audio_segments: List[AudioSegment] = []
    generated_images: List[GeneratedImage] = []
    total_audio_duration: float = 0.0
    total_generation_time: float = 0.0
    quality_metrics: Dict[str, float] = {}
    generation_stats: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_audio_file_paths(self) -> List[str]:
        """Get all audio file paths"""
        return [seg.file_path for seg in self.audio_segments if seg.file_path]
    
    def get_image_file_paths(self) -> List[str]:
        """Get all image file paths"""
        return [img.file_path for img in self.generated_images if img.file_path]
    
    def get_images_for_timeframe(self, start_time: float, end_time: float) -> List[GeneratedImage]:
        """Get images that should appear in a specific timeframe"""
        return [
            img for img in self.generated_images 
            if start_time <= img.timestamp < end_time
        ]


class VoiceProfile(BaseModel):
    """Voice profile configuration"""
    name: str
    model_path: str
    language: str = "en"
    gender: str = "neutral"
    style: str = "documentary"  # documentary, conversational, dramatic
    characteristics: Dict[str, Any] = {}
    

class StylePreset(BaseModel):
    """Image style preset"""
    name: str
    base_prompt_additions: str
    negative_prompt: str
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    style_strength: float = 0.8
    color_palette: str = ""
    mood_descriptors: str = ""
    

class MediaPipelineConfig(BaseModel):
    """Configuration for media generation pipeline"""
    # TTS Settings
    tts_engine: str = "coqui_xtts"
    default_voice: str = "documentary_narrator"
    audio_quality: AudioQuality = AudioQuality.HIGH
    
    # Image Generation Settings  
    image_engine: str = "flux"
    default_style: str = "documentary"
    image_resolution: str = "1920x1080"
    batch_size: int = 4
    
    # Hardware Settings
    device: str = "cuda"
    mixed_precision: bool = True
    memory_efficient: bool = True
    
    # Output Settings
    audio_output_dir: str = "output/audio"
    image_output_dir: str = "output/images"
    temp_dir: str = "temp/media"
    
    # Quality Settings
    audio_sample_rate: int = 44100
    audio_bit_depth: int = 16
    image_quality: int = 95
    
    # Performance Settings
    max_parallel_audio: int = 2
    max_parallel_images: int = 8
    cache_models: bool = True

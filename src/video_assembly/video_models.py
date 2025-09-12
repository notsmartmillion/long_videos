"""
Video Assembly Data Models

Pydantic models for video assembly pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum

class TransitionType(str, Enum):
    """Types of transitions between images"""
    FADE = "fade"
    DISSOLVE = "dissolve"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"

class EffectType(str, Enum):
    """Types of visual effects"""
    KEN_BURNS = "ken_burns"
    PARTICLES = "particles"
    GLOW = "glow"
    VIGNETTE = "vignette"
    CAMERA_SHAKE = "camera_shake"

class VideoQuality(str, Enum):
    """Video quality presets"""
    DRAFT = "draft"        # 720p, fast encode
    STANDARD = "standard"  # 1080p, balanced
    HIGH = "high"         # 1080p, high quality
    ULTRA = "ultra"       # 4K, maximum quality

class VideoSegment(BaseModel):
    """A single video segment with image, audio, and effects"""
    image_path: Path
    start_time: float  # seconds
    duration: float   # seconds
    audio_segment: Optional[Path] = None
    
    # Visual effects
    ken_burns_start: Tuple[float, float, float] = (1.0, 0.5, 0.5)  # zoom, x, y
    ken_burns_end: Tuple[float, float, float] = (1.2, 0.4, 0.6)
    transition_in: TransitionType = TransitionType.FADE
    transition_out: TransitionType = TransitionType.FADE
    transition_duration: float = 0.5
    
    # Effects
    effects: List[EffectType] = Field(default_factory=lambda: [EffectType.KEN_BURNS])
    particle_count: int = Field(default=30, ge=0, le=200)
    particle_speed: float = Field(default=1.0, ge=0.1, le=5.0)
    
    # Metadata
    chapter_title: Optional[str] = None
    narration_text: Optional[str] = None

class AudioTrack(BaseModel):
    """Audio track configuration"""
    file_path: Path
    volume: float = Field(default=0.8, ge=0.0, le=1.0)
    fade_in: float = Field(default=0.1, ge=0.0, le=2.0)
    fade_out: float = Field(default=0.1, ge=0.0, le=2.0)
    start_offset: float = Field(default=0.0, ge=0.0)

class VideoMetadata(BaseModel):
    """Video metadata and output settings"""
    title: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: str = "Education"
    
    # Technical settings
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = Field(default=30, ge=24, le=60)
    quality: VideoQuality = VideoQuality.STANDARD
    
    # Output
    output_path: Path
    thumbnail_path: Optional[Path] = None
    
    # Timing
    total_duration: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

class VideoAssemblyRequest(BaseModel):
    """Request for video assembly"""
    segments: List[VideoSegment]
    audio_tracks: List[AudioTrack]
    metadata: VideoMetadata
    
    # Global effects settings
    background_color: str = "#000000"
    enable_gpu_acceleration: bool = True
    enable_motion_blur: bool = True
    enable_particle_effects: bool = True
    
    # Performance settings
    max_concurrent_segments: int = Field(default=4, ge=1, le=16)
    use_preview_mode: bool = False  # Faster rendering for testing

class RenderProgress(BaseModel):
    """Progress tracking for video rendering"""
    current_segment: int = 0
    total_segments: int = 0
    current_step: str = "initializing"
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_time_remaining: Optional[float] = None  # seconds
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class VideoAssemblyResult(BaseModel):
    """Result of video assembly process"""
    success: bool
    output_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    
    # Render statistics
    total_duration: float = 0.0  # seconds
    file_size_mb: float = 0.0
    render_time_seconds: float = 0.0
    fps_actual: float = 0.0
    
    # Quality metrics
    segments_processed: int = 0
    effects_applied: int = 0
    audio_tracks_mixed: int = 0
    
    # Performance stats
    gpu_utilization_avg: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    
    # Logs and errors
    render_log: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)

class EffectSettings(BaseModel):
    """Settings for visual effects"""
    
    # Ken Burns effect
    ken_burns_intensity: float = Field(default=0.2, ge=0.0, le=1.0)
    ken_burns_speed: float = Field(default=1.0, ge=0.1, le=3.0)
    
    # Particle effects (stars)
    particle_density: float = Field(default=0.5, ge=0.0, le=2.0)
    particle_size_range: Tuple[float, float] = (1.0, 3.0)
    particle_brightness: float = Field(default=0.8, ge=0.0, le=1.0)
    particle_twinkle: bool = True
    particle_trail_length: float = Field(default=0.5, ge=0.0, le=2.0)
    
    # Glow effects
    glow_radius: float = Field(default=5.0, ge=0.0, le=20.0)
    glow_intensity: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Camera movement
    camera_drift_intensity: float = Field(default=0.1, ge=0.0, le=0.5)
    camera_shake_frequency: float = Field(default=0.0, ge=0.0, le=2.0)

class PerformanceProfile(BaseModel):
    """Performance optimization profile"""
    name: str
    description: str
    
    # Encoding settings
    encoder: str = "h264_nvenc"  # GPU-accelerated encoding
    preset: str = "p4"  # NVIDIA preset (p1-p7, p4 is balanced)
    crf: int = Field(default=23, ge=15, le=35)  # Quality (lower = better)
    
    # Processing settings
    thread_count: int = Field(default=8, ge=1, le=32)
    buffer_size_mb: int = Field(default=512, ge=128, le=2048)
    
    # GPU settings
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    enable_tensor_cores: bool = True
    enable_mixed_precision: bool = True
    
    # Quality vs speed trade-offs
    motion_estimation: str = "umh"  # me_method
    subpixel_refinement: int = Field(default=8, ge=1, le=11)
    max_b_frames: int = Field(default=3, ge=0, le=8)

# Predefined performance profiles for RTX 5080
RTX_5080_PROFILES = {
    "draft": PerformanceProfile(
        name="Draft - RTX 5080",
        description="Fastest rendering for testing and iteration",
        preset="p1",
        crf=28,
        thread_count=16,
        subpixel_refinement=4
    ),
    "balanced": PerformanceProfile(
        name="Balanced - RTX 5080", 
        description="Good quality with reasonable render times",
        preset="p4",
        crf=23,
        thread_count=12,
        subpixel_refinement=8
    ),
    "quality": PerformanceProfile(
        name="Quality - RTX 5080",
        description="High quality output, longer render times",
        preset="p6", 
        crf=20,
        thread_count=8,
        subpixel_refinement=11,
        max_b_frames=5
    )
}



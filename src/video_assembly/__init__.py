"""
Video Assembly Pipeline

This module handles the final video assembly process, combining:
- Generated audio (TTS)
- Generated images 
- Visual effects (Ken Burns, particles)
- Timing synchronization
- Final rendering

Optimized for RTX 5080 with hardware acceleration.
"""

from .video_assembler import VideoAssembler
from .video_models import VideoAssemblyRequest, VideoAssemblyResult
from .video_effects import EffectsEngine

__all__ = [
    'VideoAssembler',
    'VideoAssemblyRequest', 
    'VideoAssemblyResult',
    'EffectsEngine'
]



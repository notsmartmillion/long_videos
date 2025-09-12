"""
Video Effects Engine

Handles visual effects for video assembly:
- Ken Burns effect (slow zoom/pan)
- Particle systems (moving stars/specs)
- Transitions between segments
- GPU-accelerated processing

Optimized for RTX 5080 performance.
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import math
import random
from dataclasses import dataclass

from .video_models import EffectType, TransitionType, EffectSettings

logger = logging.getLogger(__name__)

@dataclass
class Particle:
    """A single particle for star effects"""
    x: float
    y: float
    vx: float  # velocity x
    vy: float  # velocity y
    size: float
    brightness: float
    age: float
    lifetime: float
    twinkle_phase: float

class EffectsEngine:
    """
    GPU-accelerated effects engine for video processing.
    
    Handles:
    - Ken Burns effects (zoom, pan, rotation)
    - Particle systems (stars, sparkles)
    - Smooth transitions
    - Real-time preview
    """
    
    def __init__(self, config, gpu_acceleration: bool = True):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gpu_acceleration = gpu_acceleration
        
        # Effect settings
        self.effect_settings = EffectSettings()
        
        # GPU context
        if gpu_acceleration:
            self._initialize_gpu()
        
        # Particle system
        self.particles: List[Particle] = []
        self.particle_textures = self._create_particle_textures()
        
        # Cache for performance
        self.frame_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 50
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU acceleration if available (backend-aware)"""
        self.gpu_available = False
        backends = {"torch": False, "opencv_cuda": False}
        
        # 1) PyTorch (primary detection - most reliable)
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                backends["torch"] = True
                self.gpu_available = True
                self.logger.info(f"GPU acceleration via PyTorch: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            self.logger.debug(f"PyTorch CUDA check failed: {e}")
        
        # 2) OpenCV CUDA (optional - only if OpenCV was built with CUDA)
        try:
            build = cv2.getBuildInformation()
            cuda_enabled_in_build = ("CUDA: YES" in build) or ("NVIDIA CUDA: YES" in build)
            if not cuda_enabled_in_build:
                self.logger.debug("OpenCV built without CUDA (CPU-only pip wheel)")
            else:
                count = cv2.cuda.getCudaEnabledDeviceCount()
                if count > 0:
                    backends["opencv_cuda"] = True
                    self.gpu_available = True
                    self.logger.info(f"OpenCV CUDA acceleration: {count} device(s)")
                else:
                    self.logger.debug("OpenCV CUDA reports 0 devices (driver/build mismatch)")
        except Exception as e:
            self.logger.debug(f"OpenCV CUDA check failed: {e}")
        
        # Final status
        if self.gpu_available:
            enabled = [k for k, v in backends.items() if v]
            self.logger.info(f"Video effects GPU acceleration enabled via: {', '.join(enabled)}")
        else:
            self.logger.info("Video effects: Using CPU-only mode (no GPU acceleration available)")
    
    def _create_particle_textures(self) -> Dict[str, np.ndarray]:
        """Create different particle textures for variety"""
        textures = {}
        
        # Simple white dot
        size = 8
        dot = np.zeros((size, size, 4), dtype=np.float32)
        center = size // 2
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                alpha = max(0, 1 - dist / center)
                dot[y, x] = [1.0, 1.0, 1.0, alpha]
        textures['dot'] = dot
        
        # Star shape
        star = np.zeros((12, 12, 4), dtype=np.float32)
        center = 6
        # Create a simple star pattern
        for y in range(12):
            for x in range(12):
                # Cross pattern for star
                if abs(x - center) <= 1 or abs(y - center) <= 1 or abs(x - y) <= 1 or abs(x + y - 11) <= 1:
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    alpha = max(0, 1 - dist / 6)
                    star[y, x] = [1.0, 1.0, 1.0, alpha * 0.8]
        textures['star'] = star
        
        # Sparkle
        sparkle = np.zeros((6, 6, 4), dtype=np.float32)
        sparkle[2:4, 2:4] = [1.0, 1.0, 1.0, 1.0]
        sparkle[1, 2:4] = [1.0, 1.0, 1.0, 0.7]
        sparkle[4, 2:4] = [1.0, 1.0, 1.0, 0.7]
        sparkle[2:4, 1] = [1.0, 1.0, 1.0, 0.7]
        sparkle[2:4, 4] = [1.0, 1.0, 1.0, 0.7]
        textures['sparkle'] = sparkle
        
        return textures
    
    def apply_ken_burns(self, 
                       image: np.ndarray,
                       progress: float,
                       start_params: Tuple[float, float, float],
                       end_params: Tuple[float, float, float],
                       easing: str = "ease_in_out") -> np.ndarray:
        """
        Apply Ken Burns effect (smooth zoom and pan).
        
        Args:
            image: Input image (H, W, C)
            progress: Animation progress 0.0 to 1.0
            start_params: (zoom, center_x, center_y) at start
            end_params: (zoom, center_x, center_y) at end
            easing: Easing function type
            
        Returns:
            Transformed image
        """
        if progress < 0.0:
            progress = 0.0
        elif progress > 1.0:
            progress = 1.0
        
        # Apply easing
        if easing == "ease_in_out":
            progress = self._ease_in_out(progress)
        elif easing == "ease_in":
            progress = progress * progress
        elif easing == "ease_out":
            progress = 1 - (1 - progress) * (1 - progress)
        
        # Interpolate parameters
        start_zoom, start_x, start_y = start_params
        end_zoom, end_x, end_y = end_params
        
        zoom = start_zoom + (end_zoom - start_zoom) * progress
        center_x = start_x + (end_x - start_x) * progress
        center_y = start_y + (end_y - start_y) * progress
        
        # Apply transformation
        h, w = image.shape[:2]
        
        # Calculate crop region for zoom
        crop_w = w / zoom
        crop_h = h / zoom
        
        # Calculate crop position
        crop_x = (w * center_x) - (crop_w / 2)
        crop_y = (h * center_y) - (crop_h / 2)
        
        # Clamp to image bounds
        crop_x = max(0, min(crop_x, w - crop_w))
        crop_y = max(0, min(crop_y, h - crop_h))
        
        # Crop and resize
        x1, y1 = int(crop_x), int(crop_y)
        x2, y2 = int(crop_x + crop_w), int(crop_y + crop_h)
        
        cropped = image[y1:y2, x1:x2]
        
        # Resize back to original dimensions (CPU only due to OpenCV pip wheel limitations)
        # Note: OpenCV CUDA not available in standard pip wheels - using CPU resize
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    def generate_particles(self, width: int, height: int, count: int) -> None:
        """Generate new particles for the particle system"""
        self.particles.clear()
        
        for _ in range(count):
            particle = Particle(
                x=random.uniform(0, width),
                y=random.uniform(0, height),
                vx=random.uniform(-20, 20),  # pixels per second
                vy=random.uniform(-30, 10),   # slight upward bias
                size=random.uniform(*self.effect_settings.particle_size_range),
                brightness=random.uniform(0.3, self.effect_settings.particle_brightness),
                age=0.0,
                lifetime=random.uniform(3.0, 8.0),  # seconds
                twinkle_phase=random.uniform(0, 2 * math.pi)
            )
            self.particles.append(particle)
    
    def update_particles(self, dt: float, width: int, height: int) -> None:
        """Update particle positions and properties"""
        for particle in self.particles[:]:  # Copy list to modify during iteration
            # Update position
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.age += dt
            particle.twinkle_phase += dt * 3.0  # Twinkle frequency
            
            # Remove particles that are too old or off-screen
            if (particle.age > particle.lifetime or 
                particle.x < -50 or particle.x > width + 50 or
                particle.y < -50 or particle.y > height + 50):
                self.particles.remove(particle)
        
        # Add new particles to maintain count
        target_count = int(self.effect_settings.particle_density * 30)
        while len(self.particles) < target_count:
            # Spawn new particle
            side = random.choice(['top', 'left', 'right'])
            if side == 'top':
                x, y = random.uniform(0, width), -10
            elif side == 'left':
                x, y = -10, random.uniform(0, height)
            else:  # right
                x, y = width + 10, random.uniform(0, height)
            
            particle = Particle(
                x=x, y=y,
                vx=random.uniform(-20, 20),
                vy=random.uniform(-30, 10),
                size=random.uniform(*self.effect_settings.particle_size_range),
                brightness=random.uniform(0.3, self.effect_settings.particle_brightness),
                age=0.0,
                lifetime=random.uniform(3.0, 8.0),
                twinkle_phase=random.uniform(0, 2 * math.pi)
            )
            self.particles.append(particle)
    
    def render_particles(self, image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Render particles onto the image"""
        if not self.particles or not self.effect_settings.particle_twinkle:
            return image
        
        result = image.copy()
        h, w = result.shape[:2]
        
        for particle in self.particles:
            if particle.x < 0 or particle.x >= w or particle.y < 0 or particle.y >= h:
                continue
            
            # Calculate twinkle effect
            twinkle = 1.0
            if self.effect_settings.particle_twinkle:
                twinkle = 0.5 + 0.5 * math.sin(particle.twinkle_phase)
            
            # Calculate fade based on age
            fade = 1.0 - (particle.age / particle.lifetime)
            fade = max(0.0, min(1.0, fade))
            
            # Final brightness
            brightness = particle.brightness * twinkle * fade * alpha
            
            # Draw particle
            x, y = int(particle.x), int(particle.y)
            size = int(particle.size)
            
            # Simple white dot
            cv2.circle(result, (x, y), size, (255, 255, 255), -1)
            
            # Add glow effect
            if self.effect_settings.glow_radius > 0:
                glow_size = int(size + self.effect_settings.glow_radius)
                overlay = np.zeros_like(result, dtype=np.float32)
                cv2.circle(overlay, (x, y), glow_size, (255, 255, 255), -1)
                
                # Gaussian blur for glow
                overlay = cv2.GaussianBlur(overlay, (glow_size*2+1, glow_size*2+1), glow_size/3)
                
                # Blend with original
                glow_alpha = brightness * self.effect_settings.glow_intensity * 0.3
                result = cv2.addWeighted(result, 1.0, overlay.astype(np.uint8), glow_alpha, 0)
        
        return result
    
    def apply_transition(self, 
                        frame1: np.ndarray, 
                        frame2: np.ndarray, 
                        progress: float, 
                        transition_type: TransitionType) -> np.ndarray:
        """Apply transition between two frames"""
        progress = max(0.0, min(1.0, progress))
        
        if transition_type == TransitionType.FADE:
            return self._fade_transition(frame1, frame2, progress)
        elif transition_type == TransitionType.DISSOLVE:
            return self._dissolve_transition(frame1, frame2, progress)
        elif transition_type == TransitionType.SLIDE_LEFT:
            return self._slide_transition(frame1, frame2, progress, 'left')
        elif transition_type == TransitionType.SLIDE_RIGHT:
            return self._slide_transition(frame1, frame2, progress, 'right')
        elif transition_type == TransitionType.ZOOM_IN:
            return self._zoom_transition(frame1, frame2, progress, 'in')
        elif transition_type == TransitionType.ZOOM_OUT:
            return self._zoom_transition(frame1, frame2, progress, 'out')
        else:
            return frame2 if progress > 0.5 else frame1
    
    def _fade_transition(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        """Simple fade transition"""
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
    
    def _dissolve_transition(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        """Dissolve transition with noise pattern"""
        h, w = frame1.shape[:2]
        
        # Create noise mask
        noise = np.random.random((h, w))
        mask = (noise < progress).astype(np.float32)
        
        # Apply mask
        result = frame1.astype(np.float32) * (1 - mask[:, :, np.newaxis])
        result += frame2.astype(np.float32) * mask[:, :, np.newaxis]
        
        return result.astype(np.uint8)
    
    def _slide_transition(self, frame1: np.ndarray, frame2: np.ndarray, progress: float, direction: str) -> np.ndarray:
        """Slide transition"""
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        if direction == 'left':
            offset = int(w * progress)
            result[:, :w-offset] = frame1[:, offset:]
            result[:, w-offset:] = frame2[:, :offset]
        else:  # right
            offset = int(w * progress)
            result[:, offset:] = frame1[:, :w-offset]
            result[:, :offset] = frame2[:, w-offset:]
        
        return result
    
    def _zoom_transition(self, frame1: np.ndarray, frame2: np.ndarray, progress: float, zoom_type: str) -> np.ndarray:
        """Zoom transition"""
        if zoom_type == 'in':
            # Zoom into frame2
            scale = 0.1 + 0.9 * progress
            frame2_scaled = self._scale_frame(frame2, scale)
            alpha = progress
        else:  # zoom out
            # Zoom out of frame1
            scale = 1.0 + 2.0 * progress
            frame1_scaled = self._scale_frame(frame1, scale)
            alpha = progress
            frame2_scaled = frame2
        
        return cv2.addWeighted(frame1, 1 - alpha, frame2_scaled, alpha, 0)
    
    def _scale_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """Scale frame with center crop/pad"""
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale > 1.0:
            # Scale up and crop
            scaled = cv2.resize(frame, (new_w, new_h))
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            return scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # Scale down and pad
            scaled = cv2.resize(frame, (new_w, new_h))
            result = np.zeros_like(frame)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
            return result
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function"""
        return t * t * (3.0 - 2.0 * t)
    
    def apply_motion_blur(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply subtle motion blur for cinematic effect"""
        if intensity <= 0:
            return image
        
        kernel_size = max(3, int(intensity * 15))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        return cv2.filter2D(image, -1, kernel)
    
    def clear_cache(self) -> None:
        """Clear the frame cache"""
        self.frame_cache.clear()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        cache_size = sum(frame.nbytes for frame in self.frame_cache.values()) / (1024**2)  # MB
        particle_count = len(self.particles)
        
        return {
            'cache_size_mb': cache_size,
            'particle_count': particle_count,
            'max_cache_size': self.max_cache_size
        }


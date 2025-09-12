"""
Audio Post-Processing for Authenticity

Fixes common AI-generated audio artifacts:
- Splice pops/clicks
- Inconsistent background noise
- Unnatural pauses
- Volume inconsistencies
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import logging
from scipy import signal
from pydub import AudioSegment
import tempfile

logger = logging.getLogger(__name__)

class AudioAuthenticityProcessor:
    """
    Processes AI-generated audio to sound more natural and human-recorded.
    """
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = getattr(config.tts, 'sample_rate', 22050)
        
    def process_audio_for_authenticity(self, audio_path: Path) -> Path:
        """
        Process audio to remove AI artifacts and add natural characteristics.
        
        Args:
            audio_path: Path to generated audio file
            
        Returns:
            Path to processed audio file
        """
        logger.info(f"Processing audio for authenticity: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        
        # Step 1: Remove splice pops and clicks
        audio = self._remove_splice_artifacts(audio, sr)
        
        # Step 2: Add natural background ambience
        audio = self._add_room_tone(audio, sr)
        
        # Step 3: Apply subtle compression (like a real microphone)
        audio = self._apply_natural_compression(audio)
        
        # Step 4: Add slight imperfections
        audio = self._add_natural_imperfections(audio, sr)
        
        # Step 5: Normalize and apply final processing
        audio = self._final_processing(audio)
        
        # Save processed audio
        output_path = audio_path.parent / f"{audio_path.stem}_processed{audio_path.suffix}"
        sf.write(str(output_path), audio, sr)
        
        logger.info(f"Audio processed successfully: {output_path}")
        return output_path
    
    def _remove_splice_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove pops and clicks from audio splicing"""
        
        # Detect potential splice points (sudden amplitude changes)
        diff = np.diff(audio)
        threshold = np.std(diff) * 3  # 3 standard deviations
        
        # Find splice candidates
        splice_points = np.where(np.abs(diff) > threshold)[0]
        
        # Apply crossfades at splice points
        crossfade_samples = int(0.01 * sr)  # 10ms crossfade
        
        for point in splice_points:
            start = max(0, point - crossfade_samples // 2)
            end = min(len(audio), point + crossfade_samples // 2)
            
            if end - start > 2:
                # Create smooth transition
                fade_in = np.linspace(0, 1, (end - start) // 2)
                fade_out = np.linspace(1, 0, (end - start) // 2)
                
                # Apply crossfade
                mid_point = start + len(fade_in)
                audio[start:mid_point] *= fade_in
                audio[mid_point:end] *= fade_out[:len(audio[mid_point:end])]
        
        return audio
    
    def _add_room_tone(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add subtle background ambience like a real recording"""
        
        # Generate very subtle room tone (pink noise)
        duration = len(audio) / sr
        room_tone = self._generate_pink_noise(duration, sr)
        
        # Make it very quiet (-45dB relative to speech)
        room_tone = room_tone * 0.001
        
        # Blend with original audio
        return audio + room_tone
    
    def _generate_pink_noise(self, duration: float, sr: int) -> np.ndarray:
        """Generate pink noise for room tone"""
        samples = int(duration * sr)
        
        # Generate white noise
        white_noise = np.random.normal(0, 1, samples)
        
        # Apply pink noise filter (1/f characteristic)
        b, a = signal.butter(1, 0.1, 'low')
        pink_noise = signal.filtfilt(b, a, white_noise)
        
        return pink_noise / np.max(np.abs(pink_noise))
    
    def _apply_natural_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply subtle compression like a real microphone/recording"""
        
        # Simple dynamic range compression
        threshold = 0.7
        ratio = 3.0
        
        # Find samples above threshold
        above_threshold = np.abs(audio) > threshold
        
        # Apply compression to loud parts
        compressed = audio.copy()
        compressed[above_threshold] = (
            np.sign(audio[above_threshold]) * 
            (threshold + (np.abs(audio[above_threshold]) - threshold) / ratio)
        )
        
        return compressed
    
    def _add_natural_imperfections(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add very subtle imperfections that exist in real recordings"""
        
        # Add tiny bit of harmonic distortion (very subtle)
        harmonic = audio * 0.01 * np.sin(2 * np.pi * np.arange(len(audio)) / sr * 60)
        
        # Add micro-variations in timing (pitch jitter)
        jitter = np.random.normal(0, 0.0001, len(audio))
        
        return audio + harmonic + jitter
    
    def _final_processing(self, audio: np.ndarray) -> np.ndarray:
        """Final normalization and processing"""
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        # Apply gentle high-pass filter (remove ultra-low frequencies)
        # This mimics real microphone response
        b, a = signal.butter(2, 80, 'high', fs=self.sample_rate)
        audio = signal.filtfilt(b, a, audio)
        
        return audio
    
    def process_tts_segments(self, audio_segments: List[Path]) -> Path:
        """
        Process multiple TTS segments and join them seamlessly.
        
        Args:
            audio_segments: List of audio file paths to join
            
        Returns:
            Path to seamlessly joined audio file
        """
        logger.info(f"Processing {len(audio_segments)} TTS segments for seamless joining")
        
        combined_audio = AudioSegment.empty()
        
        for i, segment_path in enumerate(audio_segments):
            segment = AudioSegment.from_file(str(segment_path))
            
            if i > 0:
                # Add natural pause between segments (200-500ms)
                pause_duration = np.random.randint(200, 500)  # Random natural pause
                silence = AudioSegment.silent(duration=pause_duration)
                
                # Apply crossfade to eliminate pops
                combined_audio = combined_audio.append(silence, crossfade=50)
                combined_audio = combined_audio.append(segment, crossfade=100)
            else:
                combined_audio = segment
        
        # Save combined audio
        output_path = audio_segments[0].parent / "combined_seamless.wav"
        combined_audio.export(str(output_path), format="wav")
        
        # Now apply authenticity processing
        return self.process_audio_for_authenticity(output_path)

class VoiceVariationProcessor:
    """
    Adds natural variations to TTS voices to avoid monotone AI detection.
    """
    
    def __init__(self, config):
        self.config = config
    
    def add_natural_variations(self, audio_path: Path) -> Path:
        """Add natural speech variations"""
        
        # Load audio with pydub for easier manipulation
        audio = AudioSegment.from_file(str(audio_path))
        
        # Apply random variations throughout
        output_audio = AudioSegment.empty()
        
        # Process in 5-10 second chunks
        chunk_size = 5000  # 5 seconds
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            
            # Randomly vary speed slightly (0.95x to 1.05x)
            speed_variation = np.random.uniform(0.98, 1.02)
            
            # Randomly vary pitch slightly (-20 to +20 cents)
            pitch_variation = np.random.uniform(-20, 20)
            
            # Apply variations
            if speed_variation != 1.0:
                chunk = chunk._spawn(chunk.raw_data, overrides={
                    "frame_rate": int(chunk.frame_rate * speed_variation)
                }).set_frame_rate(chunk.frame_rate)
            
            # Simple pitch shifting (basic implementation)
            if abs(pitch_variation) > 5:
                # Pitch shift by changing playback rate slightly
                rate_change = 2 ** (pitch_variation / 1200)  # Convert cents to ratio
                chunk = chunk._spawn(chunk.raw_data, overrides={
                    "frame_rate": int(chunk.frame_rate * rate_change)
                }).set_frame_rate(chunk.frame_rate)
            
            output_audio += chunk
        
        # Save processed audio
        output_path = audio_path.parent / f"{audio_path.stem}_varied{audio_path.suffix}"
        output_audio.export(str(output_path), format="wav")
        
        return output_path



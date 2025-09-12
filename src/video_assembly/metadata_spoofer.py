"""
Video Metadata Spoofing for Authenticity

Makes AI-generated videos appear as authentic device recordings
by spoofing metadata to mimic:
- iPhone/iPad screen recordings (ReplayKit)
- macOS QuickTime recordings  
- Android screen captures
- Real camera recordings
"""

import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
import uuid
import random

logger = logging.getLogger(__name__)

class MetadataSpoofer:
    """
    Spoofs video metadata to make AI-generated content appear authentic.
    """
    
    def __init__(self, config):
        self.config = config
        self.temp_dir = Path(getattr(config.paths, 'temp', './temp')) / 'metadata_spoofing'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def spoof_as_screen_recording(self, video_path: Path, platform: str = "ios") -> Path:
        """
        Make video appear as a screen recording from iOS, macOS, or Android.
        
        Args:
            video_path: Path to video file
            platform: "ios", "macos", or "android"
            
        Returns:
            Path to video with spoofed metadata
        """
        logger.info(f"Spoofing video metadata as {platform} screen recording: {video_path}")
        
        # Choose spoofing template based on platform
        if platform == "ios":
            return self._spoof_ios_replaykit(video_path)
        elif platform == "macos":
            return self._spoof_macos_quicktime(video_path)
        elif platform == "android":
            return self._spoof_android_screen_record(video_path)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def _spoof_ios_replaykit(self, video_path: Path) -> Path:
        """Spoof as iOS ReplayKit screen recording"""
        
        output_path = self.temp_dir / f"{video_path.stem}_ios_spoofed{video_path.suffix}"
        
        # Generate realistic iOS metadata
        creation_time = self._generate_realistic_timestamp()
        
        # FFmpeg command to re-encode with iOS-like metadata
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264',  # iOS uses H.264
            '-c:a', 'aac',   # iOS uses AAC audio
            '-movflags', '+faststart',
            
            # iOS ReplayKit specific metadata
            '-metadata', f'creation_time={creation_time}',
            '-metadata', 'com.apple.quicktime.author=ReplayKitRecording',
            '-metadata', 'com.apple.quicktime.software=iOS ReplayKit',
            '-metadata', f'com.apple.quicktime.make=Apple',
            '-metadata', f'com.apple.quicktime.model={self._random_ios_device()}',
            '-metadata', 'encoder=Core Media Video',
            
            # Video settings that match iOS
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level:v', '4.0',
            '-maxrate', '10M',
            '-bufsize', '20M',
            
            # Audio settings
            '-ar', '44100',
            '-b:a', '128k',
            
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully spoofed as iOS ReplayKit: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to spoof iOS metadata: {e.stderr}")
            raise
    
    def _spoof_macos_quicktime(self, video_path: Path) -> Path:
        """Spoof as macOS QuickTime screen recording"""
        
        output_path = self.temp_dir / f"{video_path.stem}_macos_spoofed{video_path.suffix}"
        
        creation_time = self._generate_realistic_timestamp()
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            
            # macOS QuickTime metadata
            '-metadata', f'creation_time={creation_time}',
            '-metadata', 'com.apple.quicktime.software=QuickTime Player',
            '-metadata', 'com.apple.quicktime.version=10.5',
            '-metadata', 'encoder=QuickTime',
            '-metadata', f'com.apple.quicktime.make=Apple',
            '-metadata', f'com.apple.quicktime.model={self._random_mac_model()}',
            
            # QuickTime Player screen recording settings
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-preset', 'medium',
            '-crf', '18',  # High quality like QuickTime
            
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully spoofed as macOS QuickTime: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to spoof macOS metadata: {e.stderr}")
            raise
    
    def _spoof_android_screen_record(self, video_path: Path) -> Path:
        """Spoof as Android screen recording"""
        
        output_path = self.temp_dir / f"{video_path.stem}_android_spoofed{video_path.suffix}"
        
        creation_time = self._generate_realistic_timestamp()
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264',
            '-c:a', 'aac',
            
            # Android metadata
            '-metadata', f'creation_time={creation_time}',
            '-metadata', f'com.android.capture=true',
            '-metadata', f'com.android.version={random.randint(9, 14)}',
            '-metadata', 'encoder=MediaRecorder',
            '-metadata', f'com.android.manufacturer={self._random_android_manufacturer()}',
            '-metadata', f'com.android.model={self._random_android_model()}',
            
            # Android typical encoding settings
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            '-b:v', '8M',  # Android typical bitrate
            
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully spoofed as Android screen recording: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to spoof Android metadata: {e.stderr}")
            raise
    
    def spoof_as_camera_recording(self, video_path: Path, camera_type: str = "phone") -> Path:
        """
        Make video appear as if recorded with a real camera.
        
        Args:
            video_path: Path to video file
            camera_type: "phone", "dslr", "webcam", or "camcorder"
        """
        logger.info(f"Spoofing video as {camera_type} camera recording")
        
        output_path = self.temp_dir / f"{video_path.stem}_camera_spoofed{video_path.suffix}"
        
        if camera_type == "phone":
            return self._spoof_phone_camera(video_path, output_path)
        elif camera_type == "dslr":
            return self._spoof_dslr_camera(video_path, output_path)
        elif camera_type == "webcam":
            return self._spoof_webcam(video_path, output_path)
        else:
            return self._spoof_generic_camera(video_path, output_path)
    
    def _spoof_phone_camera(self, video_path: Path, output_path: Path) -> Path:
        """Spoof as phone camera recording"""
        
        creation_time = self._generate_realistic_timestamp()
        gps_coords = self._generate_random_gps()
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            
            # Phone camera metadata
            '-metadata', f'creation_time={creation_time}',
            '-metadata', f'com.apple.quicktime.make=Apple',
            '-metadata', f'com.apple.quicktime.model={self._random_ios_device()}',
            '-metadata', f'com.apple.quicktime.software=iOS',
            '-metadata', f'location={gps_coords["lat"]},{gps_coords["lon"]}',
            '-metadata', f'location-ISO6709={gps_coords["iso6709"]}',
            '-metadata', 'com.apple.quicktime.camera.identifier=Back Camera',
            '-metadata', 'com.apple.quicktime.camera.lens_model=iPhone Camera',
            
            # Phone camera typical settings
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level:v', '4.0',
            '-b:v', '12M',  # Phone camera quality
            '-r', '30',     # 30 fps typical for phones
            
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    
    def _spoof_dslr_camera(self, video_path: Path, output_path: Path) -> Path:
        """Spoof as DSLR camera recording"""
        
        creation_time = self._generate_realistic_timestamp()
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264',
            '-c:a', 'aac',
            
            # DSLR metadata
            '-metadata', f'creation_time={creation_time}',
            '-metadata', f'make={self._random_camera_make()}',
            '-metadata', f'model={self._random_dslr_model()}',
            '-metadata', 'software=Camera Firmware',
            '-metadata', f'lens_model={self._random_lens_model()}',
            '-metadata', f'iso={random.choice([100, 200, 400, 800])}',
            '-metadata', f'exposure_time=1/{random.randint(30, 120)}',
            '-metadata', f'aperture=f/{random.choice([1.4, 1.8, 2.8, 4.0, 5.6])}',
            
            # DSLR quality settings
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-crf', '15',   # High quality
            '-preset', 'slow',
            '-r', '24',     # Cinematic frame rate
            
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    
    def _spoof_webcam(self, video_path: Path, output_path: Path) -> Path:
        """Spoof as webcam recording"""
        
        creation_time = self._generate_realistic_timestamp()
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264',
            '-c:a', 'aac',
            
            # Webcam metadata
            '-metadata', f'creation_time={creation_time}',
            '-metadata', f'encoder={self._random_webcam_software()}',
            '-metadata', f'device_name={self._random_webcam_name()}',
            '-metadata', 'source=USB Video Device',
            
            # Webcam quality (typically lower)
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            '-b:v', '2M',   # Lower bitrate
            '-r', '30',
            '-s', '1280x720',  # Common webcam resolution
            
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    
    def _generate_realistic_timestamp(self) -> str:
        """Generate a realistic creation timestamp"""
        # Random time within the last 30 days
        now = datetime.now()
        days_ago = random.randint(1, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        timestamp = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        return timestamp.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
    
    def _generate_random_gps(self) -> Dict[str, str]:
        """Generate random but realistic GPS coordinates"""
        # Random coordinates (roughly in populated areas)
        lat = random.uniform(25.0, 50.0)  # US latitude range
        lon = random.uniform(-125.0, -65.0)  # US longitude range
        
        return {
            'lat': f"{lat:.6f}",
            'lon': f"{lon:.6f}", 
            'iso6709': f"+{lat:.6f}-{abs(lon):.6f}/"
        }
    
    def _random_ios_device(self) -> str:
        """Return random iOS device model"""
        devices = [
            "iPhone 12 Pro", "iPhone 13", "iPhone 13 Pro", "iPhone 14",
            "iPhone 14 Pro", "iPhone 15", "iPhone 15 Pro", "iPad Air",
            "iPad Pro 11-inch", "iPad Pro 12.9-inch"
        ]
        return random.choice(devices)
    
    def _random_mac_model(self) -> str:
        """Return random Mac model"""
        models = [
            "MacBook Pro 16,1", "MacBook Air 13,1", "MacBook Pro 14,1",
            "iMac 21,1", "Mac Studio 13,1", "MacBook Pro 18,1"
        ]
        return random.choice(models)
    
    def _random_android_manufacturer(self) -> str:
        """Return random Android manufacturer"""
        return random.choice(["Samsung", "Google", "OnePlus", "Xiaomi", "Sony"])
    
    def _random_android_model(self) -> str:
        """Return random Android model"""
        models = [
            "Galaxy S23", "Pixel 7", "OnePlus 11", "Xiaomi 13",
            "Galaxy Note 20", "Pixel 6 Pro", "OnePlus 9 Pro"
        ]
        return random.choice(models)
    
    def _random_camera_make(self) -> str:
        """Return random camera manufacturer"""
        return random.choice(["Canon", "Nikon", "Sony", "Fujifilm", "Panasonic"])
    
    def _random_dslr_model(self) -> str:
        """Return random DSLR model"""
        models = [
            "EOS R5", "EOS R6", "D850", "D780", "A7 IV", 
            "A7R V", "X-T5", "GH6", "Z9", "R7"
        ]
        return random.choice(models)
    
    def _random_lens_model(self) -> str:
        """Return random lens model"""
        lenses = [
            "24-70mm f/2.8", "50mm f/1.8", "85mm f/1.4", "24-105mm f/4",
            "70-200mm f/2.8", "35mm f/1.4", "16-35mm f/2.8"
        ]
        return random.choice(lenses)
    
    def _random_webcam_software(self) -> str:
        """Return random webcam software"""
        return random.choice(["OBS Studio", "Zoom", "Teams", "Chrome", "QuickTime"])
    
    def _random_webcam_name(self) -> str:
        """Return random webcam device name"""
        names = [
            "Logitech HD Pro Webcam C920", "Microsoft LifeCam HD-3000",
            "Razer Kiyo", "FaceTime HD Camera", "USB 2.0 Camera"
        ]
        return random.choice(names)
    
    def _spoof_generic_camera(self, video_path: Path, output_path: Path) -> Path:
        """Generic camera spoofing"""
        creation_time = self._generate_realistic_timestamp()
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'h264', '-c:a', 'aac',
            '-metadata', f'creation_time={creation_time}',
            '-metadata', 'encoder=Camera',
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    
    def apply_device_compression_artifacts(self, video_path: Path, device_type: str = "phone") -> Path:
        """
        Apply compression artifacts typical of different devices.
        This makes the video look like it went through device processing.
        """
        output_path = self.temp_dir / f"{video_path.stem}_compressed{video_path.suffix}"
        
        if device_type == "phone":
            # Phone cameras typically have some compression
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-c:v', 'h264',
                '-preset', 'fast',  # Phone processing is fast
                '-crf', '23',       # Slight compression
                '-b:v', '8M',       # Phone bitrate
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path)
            ]
        elif device_type == "webcam":
            # Webcams have more compression, lower quality
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-c:v', 'h264',
                '-preset', 'veryfast',
                '-crf', '28',       # More compression
                '-b:v', '2M',       # Lower bitrate
                '-s', '1280x720',   # Lower resolution
                '-c:a', 'aac', '-b:a', '96k',
                str(output_path)
            ]
        else:
            # Generic device compression
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-c:v', 'h264',
                '-preset', 'medium',
                '-crf', '25',
                '-c:a', 'aac',
                str(output_path)
            ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Applied {device_type} compression artifacts: {output_path}")
        return output_path
    
    def cleanup_temp_files(self):
        """Clean up temporary spoofed files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)



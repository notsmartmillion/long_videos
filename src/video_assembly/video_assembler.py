"""
Video Assembler

Main video assembly engine that combines all components:
- Audio synchronization
- Image sequencing with timing
- Visual effects application
- GPU-accelerated rendering
- Final video output

Optimized for RTX 5080 with CUDA acceleration.
"""

import asyncio
import logging
import time
import subprocess
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
import cv2
import numpy as np
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock

from ..content_generation.content_models import VideoScript
from .video_models import (
    VideoAssemblyRequest, VideoAssemblyResult, VideoSegment, 
    VideoMetadata, AudioTrack, RenderProgress, PerformanceProfile,
    RTX_5080_PROFILES, VideoQuality, EffectType
)
from .video_effects import EffectsEngine
from .audio_processor import AudioAuthenticityProcessor
from .metadata_spoofer import MetadataSpoofer

class VideoAssembler:
    """
    High-performance video assembler optimized for RTX 5080.
    
    Features:
    - GPU-accelerated rendering with NVENC
    - Parallel segment processing
    - Real-time effects application
    - Automatic timing synchronization
    - Memory-efficient streaming
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.output_dir = Path(getattr(config.paths, 'output', './output')) / 'videos'
        self.temp_dir = Path(getattr(config.paths, 'temp', './temp')) / 'video_assembly'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Effects engine
        self.effects_engine = EffectsEngine(config, gpu_acceleration=True)
        
        # Authenticity processors
        self.audio_processor = AudioAuthenticityProcessor(config)
        self.metadata_spoofer = MetadataSpoofer(config)
        
        # Performance settings
        self.max_workers = getattr(config.performance, 'max_parallel_video', 4)
        self.gpu_memory_fraction = getattr(config.performance, 'gpu_memory_fraction', 0.8)
        
        # Render state
        self.current_render: Optional[RenderProgress] = None
        self.render_lock = Lock()
        
        # Executors for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Check FFmpeg and GPU capabilities
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available"""
        try:
            # Check FFmpeg
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.ffmpeg_available = True
                # Check for NVENC support
                self.nvenc_available = 'nvenc' in result.stdout.lower()
                self.logger.info(f"FFmpeg available, NVENC: {self.nvenc_available}")
            else:
                self.ffmpeg_available = False
                self.nvenc_available = False
                self.logger.error("FFmpeg not found - video rendering will not work")
        except Exception as e:
            self.ffmpeg_available = False
            self.nvenc_available = False
            self.logger.error(f"Failed to check FFmpeg: {e}")
    
    async def assemble_video(self, 
                           video_script: VideoScript,
                           audio_path: Path,
                           image_paths: List[Path],
                           topic_category: str,
                           progress_callback: Optional[Callable[[RenderProgress], None]] = None,
                           timeline_path: Optional[str] = None) -> VideoAssemblyResult:
        """
        Assemble a complete video from generated content.
        
        Args:
            video_script: Generated script with timing and metadata
            audio_path: Path to generated audio file
            image_paths: List of generated image paths
            topic_category: Topic category for styling
            progress_callback: Optional callback for progress updates
            
        Returns:
            VideoAssemblyResult with output path and statistics
        """
        start_time = time.time()
        
        try:
            # Create assembly request
            if timeline_path and Path(timeline_path).exists():
                request = await self._create_assembly_request_from_timeline(
                    video_script, audio_path, topic_category, timeline_path
                )
            else:
                request = await self._create_assembly_request(
                    video_script, audio_path, image_paths, topic_category
                )
            
            # Initialize progress tracking
            progress = RenderProgress(
                total_segments=len(request.segments),
                current_step="initializing"
            )
            
            with self.render_lock:
                self.current_render = progress
            
            if progress_callback:
                progress_callback(progress)
            
            # Process video segments
            self.logger.info(f"Starting video assembly with {len(request.segments)} segments")
            processed_segments = await self._process_segments(request, progress, progress_callback)
            
            # Render final video
            progress.current_step = "rendering_final_video"
            progress.progress_percent = 80.0
            if progress_callback:
                progress_callback(progress)
            
            output_path = await self._render_final_video(request, processed_segments, progress)
            
            # Apply authenticity processing
            progress.current_step = "applying_authenticity_processing"
            progress.progress_percent = 90.0
            if progress_callback:
                progress_callback(progress)
            
            authentic_video_path = await self._apply_authenticity_processing(output_path, request)
            
            # Create thumbnail
            progress.current_step = "generating_thumbnail"
            progress.progress_percent = 95.0
            if progress_callback:
                progress_callback(progress)
            
            thumbnail_path = await self._generate_thumbnail(authentic_video_path, request.metadata)
            
            # Calculate final statistics
            render_time = time.time() - start_time
            file_size = output_path.stat().st_size / (1024**2) if output_path.exists() else 0.0
            
            # Create result
            result = VideoAssemblyResult(
                success=True,
                output_path=authentic_video_path,
                thumbnail_path=thumbnail_path,
                total_duration=request.metadata.total_duration or 0.0,
                file_size_mb=file_size,
                render_time_seconds=render_time,
                segments_processed=len(processed_segments),
                effects_applied=sum(len(seg.effects) for seg in request.segments),
                audio_tracks_mixed=len(request.audio_tracks)
            )
            
            progress.current_step = "completed"
            progress.progress_percent = 100.0
            if progress_callback:
                progress_callback(progress)
            
            self.logger.info(f"Video assembly completed in {render_time:.1f}s: {output_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Video assembly failed: {e}")
            return VideoAssemblyResult(
                success=False,
                errors=[str(e)],
                render_time_seconds=time.time() - start_time
            )
        finally:
            with self.render_lock:
                self.current_render = None
    
    async def _create_assembly_request(self,
                                     video_script: VideoScript,
                                     audio_path: Path,
                                     image_paths: List[Path],
                                     topic_category: str) -> VideoAssemblyRequest:
        """Create video assembly request from generated content"""
        
        # Get audio duration first - this determines video length
        audio_duration = await self._get_audio_duration(audio_path)
        self.logger.info(f"🎵 Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
        
        # Calculate timing for image segments based on audio duration
        if len(image_paths) > 0:
            segment_duration = audio_duration / len(image_paths)  # Distribute evenly across audio
            self.logger.info(f"🖼️ {len(image_paths)} images, {segment_duration:.1f}s per image")
        else:
            segment_duration = 8.0  # Fallback
        
        # Create segments
        segments = []
        total_time = 0.0
        
        for i, (image_path, image_prompt) in enumerate(zip(image_paths, video_script.image_prompts)):
            # Calculate Ken Burns parameters for variety
            start_zoom = 1.0 + (i % 3) * 0.1  # Vary starting zoom
            end_zoom = start_zoom + 0.2
            
            start_x = 0.4 + (i % 5) * 0.05  # Vary pan start
            start_y = 0.4 + (i % 3) * 0.1
            end_x = 0.6 - (i % 5) * 0.05   # Vary pan end
            end_y = 0.6 - (i % 3) * 0.1
            
            segment = VideoSegment(
                image_path=image_path,
                start_time=total_time,
                duration=segment_duration,
                ken_burns_start=(1.0, 0.5, 0.5),  # Static parameters (not used)
                ken_burns_end=(1.0, 0.5, 0.5),    # Static parameters (not used)
                effects=[],  # No effects - clean static images only
                particle_count=0,  # No particles
                chapter_title=getattr(image_prompt, 'chapter_title', None),
                narration_text=getattr(image_prompt, 'description', '')
            )
            segments.append(segment)
            total_time += segment_duration
        
        # Audio track - use proportional fade based on duration
        fade_duration = min(0.5, audio_duration * 0.02)  # Max 0.5s or 2% of duration
        audio_tracks = [AudioTrack(
            file_path=audio_path,
            volume=0.85,
            fade_in=fade_duration,
            fade_out=fade_duration
        )]
        
        # Video metadata - use audio duration as the definitive video length
        safe_category = str(topic_category) if topic_category else "generic"
        safe_subtopic = getattr(video_script.metadata, 'subtopic', None) or 'general'
        output_filename = f"{safe_category}_{safe_subtopic}_{int(time.time())}.mp4"
        metadata = VideoMetadata(
            title=video_script.title,
            description=video_script.description,
            tags=getattr(video_script.metadata, 'tags', []),
            category=safe_category.title(),
            total_duration=audio_duration,  # Video length = audio length
            output_path=self.output_dir / output_filename,
            quality=VideoQuality.HIGH  # Use high quality for RTX 5080
        )
        
        self.logger.info(f"🎬 Video will be {audio_duration:.1f}s long to match audio")
        
        return VideoAssemblyRequest(
            segments=segments,
            audio_tracks=audio_tracks,
            metadata=metadata,
            enable_gpu_acceleration=True,
            enable_motion_blur=True,
            enable_particle_effects=True,
            max_concurrent_segments=4
        )

    async def _create_assembly_request_from_timeline(self,
                                     video_script: VideoScript,
                                     audio_path: Path,
                                     topic_category: str,
                                     timeline_path: str) -> VideoAssemblyRequest:
        import json
        data = json.loads(Path(timeline_path).read_text(encoding='utf-8'))
        beats = data.get('beats', [])
        # Build segments from beats
        segments = []
        for b in beats:
            image_path = b.get('chosen_image')
            start_s = float(b.get('start_s', 0.0))
            end_s = float(b.get('end_s', start_s + 8.0))
            duration = max(0.1, end_s - start_s)
            segments.append(VideoSegment(
                image_path=Path(image_path),
                start_time=start_s,
                duration=duration,
                ken_burns_start=(1.0, 0.5, 0.5),
                ken_burns_end=(1.0, 0.5, 0.5),
                effects=[],
                particle_count=0,
                chapter_title=None,
                narration_text=""
            ))
        # Audio track
        audio_tracks = [AudioTrack(file_path=audio_path, volume=0.85, fade_in=0.3, fade_out=0.3)]
        # Metadata
        total_duration = segments[-1].start_time + segments[-1].duration if segments else await self._get_audio_duration(audio_path)
        safe_category = str(topic_category) if topic_category else "generic"
        safe_subtopic = getattr(video_script.metadata, 'subtopic', None) or 'general'
        output_filename = f"{safe_category}_{safe_subtopic}_{int(time.time())}.mp4"
        metadata = VideoMetadata(
            title=video_script.title,
            description=video_script.description,
            tags=getattr(video_script.metadata, 'tags', []),
            category=safe_category.title(),
            total_duration=total_duration,
            output_path=self.output_dir / output_filename,
            quality=VideoQuality.HIGH
        )
        return VideoAssemblyRequest(
            segments=segments,
            audio_tracks=audio_tracks,
            metadata=metadata,
            enable_gpu_acceleration=True,
            enable_motion_blur=True,
            enable_particle_effects=True,
            max_concurrent_segments=4
        )
    
    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio file duration using ffprobe"""
        try:
            probe = ffmpeg.probe(str(audio_path))
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            self.logger.warning(f"Could not get audio duration: {e}")
            return 120.0  # Default fallback
    
    async def _process_segments(self,
                              request: VideoAssemblyRequest,
                              progress: RenderProgress,
                              progress_callback: Optional[Callable[[RenderProgress], None]]) -> List[Path]:
        """Process all video segments with effects"""
        
        processed_paths = []
        
        # Calculate total frames for more accurate progress
        total_frames = sum(int(seg.duration * request.metadata.fps) for seg in request.segments)
        frames_processed = 0
        
        # Process segments sequentially for better progress reporting
        for i, segment in enumerate(request.segments):
            try:
                # Update progress at segment start
                progress.current_step = f"processing_segment_{i+1}_of_{len(request.segments)}"
                progress.current_segment = i + 1
                segment_progress = (i / len(request.segments)) * 70.0
                progress.progress_percent = segment_progress
                
                if progress_callback:
                    progress_callback(progress)
                
                # Process segment with frame-level progress
                def frame_progress_callback(frame_num: int, total_segment_frames: int):
                    nonlocal frames_processed
                    frames_processed += 1
                    # Update progress every 10 frames to avoid spam
                    if frame_num % 10 == 0 or frame_num == total_segment_frames - 1:
                        frame_progress = (frames_processed / total_frames) * 70.0
                        progress.progress_percent = frame_progress
                        progress.current_step = f"segment_{i+1}_frame_{frame_num+1}_of_{total_segment_frames}"
                        if progress_callback:
                            progress_callback(progress)
                
                result = await self._process_single_segment(segment, request, i, frame_progress_callback)
                processed_paths.append(result)
                
            except Exception as e:
                self.logger.error(f"Segment {i} failed: {e}")
                progress.errors.append(f"Segment {i}: {e}")
                # Continue with other segments
        
        return processed_paths
    
    async def _process_single_segment(self,
                                    segment: VideoSegment,
                                    request: VideoAssemblyRequest,
                                    segment_index: int,
                                    frame_progress_callback: Optional[Callable[[int, int], None]] = None) -> Path:
        """Process a single video segment with effects"""
        
        # Load image
        image = cv2.imread(str(segment.image_path))
        if image is None:
            raise ValueError(f"Could not load image: {segment.image_path}")
        
        # Resize to target resolution
        target_w, target_h = request.metadata.resolution
        image = cv2.resize(image, (target_w, target_h))
        
        # Create video frames for this segment
        fps = request.metadata.fps
        total_frames = int(segment.duration * fps)
        
        # Output path for this segment
        segment_output = self.temp_dir / f"segment_{segment_index:04d}.mp4"
        
        # Initialize effects
        self.effects_engine.generate_particles(target_w, target_h, segment.particle_count)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(segment_output), fourcc, fps, (target_w, target_h))
        
        # Use fast FFmpeg-based processing instead of slow frame-by-frame
        writer.release()  # Close the unused writer
        
        # Always use static images (Ken Burns effects removed per user request)
        await self._create_static_video_with_ffmpeg(
            segment.image_path,
            segment_output,
            segment.duration,
            fps,
            total_frames,
            frame_progress_callback
        )
        
        return segment_output

    async def _create_ken_burns_with_ffmpeg(self,
                                          image_path: Path,
                                          output_path: Path,
                                          duration: float,
                                          fps: int,
                                          start_params: Tuple[float, float, float],
                                          end_params: Tuple[float, float, float],
                                          total_frames: int,
                                          frame_progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """Create Ken Burns effect using FFmpeg hardware acceleration (ultra-fast)"""
        
        start_zoom, start_x, start_y = start_params
        end_zoom, end_x, end_y = end_params
        
        # FFmpeg zoompan filter for hardware-accelerated Ken Burns
        zoompan_filter = (
            f"zoompan="
            f"z='if(eq(on,1),{start_zoom},{start_zoom}+({end_zoom}-{start_zoom})*(on-1)/({total_frames}-1))':"
            f"x='if(eq(on,1),{start_x}*iw,{start_x}*iw+({end_x}-{start_x})*iw*(on-1)/({total_frames}-1))':"
            f"y='if(eq(on,1),{start_y}*ih,{start_y}*ih+({end_y}-{start_y})*ih*(on-1)/({total_frames}-1))':"
            f"d={total_frames}:s=1920x1080:fps={fps}"
        )
        
        # Use NVENC hardware encoding for RTX 5080
        cmd = [
            'ffmpeg', '-y',
            '-i', str(image_path),
            '-vf', zoompan_filter,
            '-c:v', 'h264_nvenc',  # RTX 5080 hardware encoding
            '-preset', 'fast',
            '-b:v', '8M',
            '-pix_fmt', 'yuv420p',
            '-t', str(duration),
            str(output_path)
        ]
        
        self.logger.info(f"🚀 FFmpeg Ken Burns: {duration}s video with RTX 5080 acceleration")
        
        # Run FFmpeg asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Simulate progress reporting (FFmpeg is much faster than frame-by-frame)
        if frame_progress_callback:
            # Report progress in chunks while FFmpeg runs
            progress_task = asyncio.create_task(self._simulate_ffmpeg_progress(total_frames, frame_progress_callback, duration))
        
        stdout, stderr = await process.communicate()
        
        if frame_progress_callback:
            progress_task.cancel()
            frame_progress_callback(total_frames, total_frames)  # Complete
        
        if process.returncode != 0:
            self.logger.error(f"FFmpeg Ken Burns failed: {stderr.decode()}")
            raise RuntimeError(f"FFmpeg Ken Burns failed: {stderr.decode()}")
        
        self.logger.info("✅ FFmpeg Ken Burns completed successfully")

    async def _create_static_video_with_ffmpeg(self,
                                             image_path: Path,
                                             output_path: Path,
                                             duration: float,
                                             fps: int,
                                             total_frames: int,
                                             frame_progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """Create static video from image using FFmpeg (very fast)"""
        
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', str(image_path),
            '-c:v', 'h264_nvenc',  # RTX 5080 hardware encoding
            '-preset', 'fast',
            '-b:v', '5M',
            '-vf', 'scale=1920:1080',
            '-pix_fmt', 'yuv420p',
            '-t', str(duration),
            str(output_path)
        ]
        
        self.logger.info(f"🚀 FFmpeg static video: {duration}s with RTX 5080 acceleration")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Quick progress simulation for static videos (very fast)
        if frame_progress_callback:
            progress_task = asyncio.create_task(self._simulate_ffmpeg_progress(total_frames, frame_progress_callback, duration))
        
        stdout, stderr = await process.communicate()
        
        if frame_progress_callback:
            progress_task.cancel()
            frame_progress_callback(total_frames, total_frames)
        
        if process.returncode != 0:
            self.logger.error(f"FFmpeg static video failed: {stderr.decode()}")
            raise RuntimeError(f"FFmpeg static video failed: {stderr.decode()}")
        
        self.logger.info("✅ FFmpeg static video completed successfully")

    async def _simulate_ffmpeg_progress(self, total_frames: int, callback: Callable[[int, int], None], duration: float):
        """Simulate progress reporting for FFmpeg operations"""
        try:
            # Update progress every 10% for smooth UI updates
            steps = 10
            for i in range(steps + 1):
                progress_frames = int((i / steps) * total_frames)
                callback(progress_frames, total_frames)
                await asyncio.sleep(duration / (steps * 2))  # FFmpeg is much faster than real-time
        except asyncio.CancelledError:
            pass
    
    async def _render_final_video(self,
                                request: VideoAssemblyRequest,
                                segment_paths: List[Path],
                                progress: RenderProgress) -> Path:
        """Render final video with audio using FFmpeg"""
        
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg not available for video rendering")
        
        # Create concat file for FFmpeg
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for path in segment_paths:
                f.write(f"file '{path.absolute()}'\n")
        
        # Performance profile for RTX 5080
        profile = RTX_5080_PROFILES.get(request.metadata.quality.value, RTX_5080_PROFILES['balanced'])
        
        # Build FFmpeg command
        output_path = request.metadata.output_path
        
        # Input streams
        inputs = []
        inputs.append(ffmpeg.input(str(concat_file), format='concat', safe=0))  # Video segments
        
        # Add audio tracks
        for audio_track in request.audio_tracks:
            audio_input = ffmpeg.input(str(audio_track.file_path))
            inputs.append(audio_input)
        
        # Video processing
        video = inputs[0]
        
        # Audio processing (mix if multiple tracks)
        if len(request.audio_tracks) == 1:
            audio = inputs[1]
            # Apply volume and fades
            track = request.audio_tracks[0]
            if track.volume != 1.0:
                audio = audio.filter('volume', track.volume)

            # Fade-in at start
            if track.fade_in > 0:
                audio = audio.filter('afade', type='in', duration=track.fade_in)

            # Correct fade-out: start near the end of audio instead of at t=0
            if track.fade_out > 0:
                # We have total duration in request.metadata.total_duration
                total_dur = max(0.0, float(getattr(request.metadata, 'total_duration', 0.0)))
                # Compute fade-out start time (st)
                st = max(0.0, total_dur - float(track.fade_out)) if total_dur > 0 else 0.0
                # Add small guard so fade-out never overlaps fade-in
                if track.fade_in > 0 and st < (track.fade_in + 0.05):
                    st = track.fade_in + 0.05

                # Log the computed fade timings for debugging
                self.logger.info(
                    f"Audio fade config -> volume={track.volume}, fade_in={track.fade_in:.2f}s, "
                    f"fade_out={track.fade_out:.2f}s, total={total_dur:.2f}s, fade_out_start={st:.2f}s"
                )

                # Apply fade-out with explicit start time
                audio = audio.filter('afade', type='out', st=st, d=track.fade_out)
        else:
            # Mix multiple audio tracks
            audio_filters = []
            for i, track in enumerate(request.audio_tracks):
                audio_stream = inputs[i + 1]
                if track.volume != 1.0:
                    audio_stream = audio_stream.filter('volume', track.volume)
                audio_filters.append(audio_stream)
            
            audio = ffmpeg.filter(audio_filters, 'amix', inputs=len(audio_filters))
        
        # Output configuration
        output_args = {
            'vcodec': profile.encoder if self.nvenc_available else 'libx264',
            'acodec': 'aac',
            'audio_bitrate': '128k',
            'preset': profile.preset if self.nvenc_available else 'medium',
            'crf': profile.crf,
            'pix_fmt': 'yuv420p',
            'movflags': '+faststart'  # Enable streaming
        }
        
        # GPU-specific options
        if self.nvenc_available:
            output_args.update({
                'gpu': 0,
                'rc': 'vbr',
                'rc_lookahead': 20,
                'surfaces': 8,
                'bf': profile.max_b_frames
            })
        
        # Build and run FFmpeg command
        try:
            (
                ffmpeg
                .output(video, audio, str(output_path), **output_args)
                .overwrite_output()
                .run(quiet=False, capture_stdout=True)
            )
            
            if not output_path.exists():
                raise RuntimeError("FFmpeg did not create output file")
            
            self.logger.info(f"Final video rendered: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"Video rendering failed: {error_msg}")
    
    async def _generate_thumbnail(self, video_path: Path, metadata: VideoMetadata) -> Optional[Path]:
        """Generate video thumbnail"""
        try:
            thumbnail_path = video_path.with_suffix('.jpg')
            
            # Extract frame at 10% of video duration
            (
                ffmpeg
                .input(str(video_path), ss=metadata.total_duration * 0.1)
                .output(str(thumbnail_path), vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(quiet=True)
            )
            
            return thumbnail_path if thumbnail_path.exists() else None
            
        except Exception as e:
            self.logger.warning(f"Thumbnail generation failed: {e}")
            return None
    
    def get_render_status(self) -> Optional[RenderProgress]:
        """Get current render progress"""
        with self.render_lock:
            return self.current_render
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files"""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to clean temp files: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'ffmpeg_available': self.ffmpeg_available,
            'nvenc_available': self.nvenc_available,
            'max_workers': self.max_workers,
            'effects_memory': self.effects_engine.get_memory_usage(),
            'temp_files': len(list(self.temp_dir.glob("*")))
        }
    
    async def _apply_authenticity_processing(self, video_path: Path, request: VideoAssemblyRequest) -> Path:
        """Apply authenticity processing to make video appear non-AI generated"""
        
        # Get authenticity settings from config
        authenticity_config = getattr(self.config, 'authenticity', {})
        
        if not authenticity_config.get('enabled', False):
            return video_path
        
        try:
            # Step 1: Extract and process audio for authenticity
            if authenticity_config.get('process_audio', True):
                temp_audio = self.temp_dir / "temp_audio.wav"
                
                # Extract audio
                (
                    ffmpeg
                    .input(str(video_path))
                    .output(str(temp_audio), acodec='pcm_s16le')
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Process audio for authenticity
                processed_audio = self.audio_processor.process_audio_for_authenticity(temp_audio)
                
                # Replace audio in video
                temp_video_with_audio = self.temp_dir / f"{video_path.stem}_with_processed_audio.mp4"
                (
                    ffmpeg
                    .input(str(video_path))
                    .input(str(processed_audio))
                    .output(str(temp_video_with_audio), vcodec='copy', acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                video_path = temp_video_with_audio
            
            # Step 2: Apply device compression artifacts
            if authenticity_config.get('apply_device_compression', True):
                device_type = authenticity_config.get('device_type', 'phone')
                video_path = self.metadata_spoofer.apply_device_compression_artifacts(video_path, device_type)
            
            # Step 3: Spoof metadata
            if authenticity_config.get('spoof_metadata', True):
                spoof_type = authenticity_config.get('spoof_as', 'ios')  # ios, macos, android, phone_camera
                
                if spoof_type in ['ios', 'macos', 'android']:
                    video_path = self.metadata_spoofer.spoof_as_screen_recording(video_path, spoof_type)
                else:
                    video_path = self.metadata_spoofer.spoof_as_camera_recording(video_path, spoof_type)
            
            self.logger.info(f"Authenticity processing completed: {video_path}")
            return video_path
            
        except Exception as e:
            self.logger.warning(f"Authenticity processing failed, using original video: {e}")
            return video_path
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
        except:
            pass

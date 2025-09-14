"""Text-to-Speech engine using Coqui XTTS v2 for high-quality local voice synthesis"""

import asyncio
import logging
import os
import subprocess
import sys
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
from src.utils.text_normalize import normalize_name_possessives

from .media_models import AudioSegment as AudioSegmentModel, AudioGenerationRequest, VoiceProfile


class TTSEngine:
    """High-quality TTS engine using Coqui XTTS v2"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('video_ai.tts_engine')
        
        # Paths
        self.output_dir = Path(getattr(config.paths, 'output', './output')) / 'audio'
        self.temp_dir = Path(getattr(config.paths, 'temp', './temp')) / 'audio'
        self.models_dir = Path(getattr(config.paths, 'models', './models')) / 'tts'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # TTS settings
        self.device = config.system.device
        self.sample_rate = 22050  # XTTS default
        # Engine preference: allow forcing SAPI for prod runs
        self.engine_name = str(getattr(config.tts, 'engine', 'coqui_xtts')).lower()
        self.prefer_sapi = self.engine_name in {"sapi", "windows_sapi", "win_sapi"}
        
        # Optimize chunk size and mode based on video length (for test mode)
        video_length = getattr(config.content, 'video_length_minutes', 60)
        if video_length <= 5:
            self.chunk_size = 500  # Smaller chunks for test mode
            self.test_mode = True  # Use Windows SAPI (fast) instead of XTTS (slow)
            self.logger.info("TTS Test mode enabled: Using fast Windows SAPI instead of XTTS")
        else:
            self.chunk_size = 1000  # Default chunk size
            self.test_mode = False
        if self.prefer_sapi and not self.test_mode:
            self.logger.info("TTS engine set to SAPI via config: forcing Windows SAPI for production run")
        
        # Voice profiles
        self.voice_profiles = self._load_voice_profiles()
        
        # Model state
        self.model = None
        self.is_loaded = False
        
    def _load_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Load available voice profiles"""
        profiles = {
            "documentary_narrator": VoiceProfile(
                name="Documentary Narrator",
                model_path="xtts_v2",
                language="en",
                gender="neutral",
                style="documentary",
                characteristics={
                    "pace": "measured",
                    "tone": "authoritative",
                    "clarity": "high"
                }
            ),
            "storyteller": VoiceProfile(
                name="Storyteller",
                model_path="xtts_v2",
                language="en", 
                gender="neutral",
                style="narrative",
                characteristics={
                    "pace": "engaging",
                    "tone": "warm",
                    "expression": "dynamic"
                }
            ),
            "educational": VoiceProfile(
                name="Educational",
                model_path="xtts_v2",
                language="en",
                gender="neutral", 
                style="educational",
                characteristics={
                    "pace": "clear",
                    "tone": "friendly",
                    "emphasis": "explanatory"
                }
            ),
            "soft_philosopher": VoiceProfile(
                name="Soft Philosopher",
                model_path="xtts_v2",
                language="en",
                gender="neutral",
                style="soft_philosophy",
                characteristics={
                    "pace": "slow",
                    "tone": "gentle",
                    "expression": "soothing",
                    "clarity": "high"
                }
            )
        }
        return profiles
    
    async def initialize(self):
        """Initialize the TTS model"""
        if self.is_loaded:
            return
            
        try:
            self.logger.info("Initializing Coqui XTTS v2 model...")
            
            # Check if XTTS is available
            if not self._check_xtts_installation():
                await self._install_xtts()
            
            # Load model (this will be done when needed to save memory)
            self.logger.info("TTS engine initialized successfully")
            self.is_loaded = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    async def generate_audio(self, request: AudioGenerationRequest, 
                           topic: Optional[str] = None, progress_callback=None) -> List[AudioSegmentModel]:
        """Generate audio from text"""
        
        if not self.is_loaded:
            await self.initialize()
        
        self.logger.info(f"Generating audio: {len(request.script_text)} characters")
        
        try:
            # Normalize possessives before any chunking
            try:
                request_text = normalize_name_possessives(request.script_text)
            except Exception:
                request_text = request.script_text

            # Split text into manageable chunks
            text_chunks = self._split_text_into_chunks(request_text)
            
            # Generate audio for each chunk
            audio_segments = []
            cumulative_time = 0.0
            
            for i, chunk in enumerate(text_chunks):
                # Update progress (only every 5% to reduce spam)
                chunk_progress = (i / len(text_chunks)) * 80  # 80% for chunks, 20% for concatenation
                if progress_callback and (i == 0 or chunk_progress - getattr(self, '_last_chunk_progress', 0) >= 5):
                    progress_callback(chunk_progress, f"Processing audio chunk {i+1}/{len(text_chunks)}")
                    self._last_chunk_progress = chunk_progress
                
                # Only log every 5th chunk to reduce spam
                if i % 5 == 0 or i == len(text_chunks) - 1:
                    self.logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                
                # Generate audio for this chunk
                audio_data, duration = await self._generate_chunk_audio(
                    chunk, request, f"segment_{i:03d}"
                )
                
                segment = AudioSegmentModel(
                    id=f"segment_{i:03d}",
                    text=chunk,
                    file_path=audio_data,
                    start_time=cumulative_time,
                    duration=duration,
                    speaker_voice=request.voice_model,
                    processing_settings={
                        "quality": request.quality.value,
                        "speed": request.speed,
                        "pitch": request.pitch,
                        "volume": request.volume
                    }
                )
                
                audio_segments.append(segment)
                cumulative_time += duration
            
            # Final progress update
            if progress_callback:
                progress_callback(100, f"Generated {len(audio_segments)} audio segments")
            
            self.logger.info(f"Generated {len(audio_segments)} audio segments, total duration: {cumulative_time:.1f}s")
            return audio_segments
            
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            raise
    
    async def _generate_chunk_audio(self, text: str, request: AudioGenerationRequest, 
                                  segment_id: str) -> tuple[str, float]:
        """Generate audio for a single text chunk"""
        
        try:
            # Output file path
            output_file = self.temp_dir / f"{segment_id}.wav"
            
            # Log the text being generated for debugging
            self.logger.info(f"🎵 Generating TTS for text: '{text[:100]}...' (length: {len(text)} chars)")
            
            # Use different TTS methods based on available engines and mode
            if self.prefer_sapi:
                # Force Windows SAPI regardless of test_mode
                self.logger.info(f"Config forced SAPI: Using Windows SAPI for chunk {segment_id}")
                duration = await self._generate_with_sapi(text, output_file, request)
            elif self.test_mode:
                # Use faster fallback method in test mode
                self.logger.info(f"Test mode: Using fast SAPI TTS for chunk {segment_id}")
                duration = await self._generate_with_fallback(text, output_file, request)
            elif self._is_coqui_available():
                self.logger.info(f"Using Coqui XTTS for chunk {segment_id}")
                duration = await self._generate_with_coqui(text, output_file, request)
            else:
                # Fallback to system TTS or other engines
                self.logger.info(f"Using fallback TTS for chunk {segment_id}")
                duration = await self._generate_with_fallback(text, output_file, request)
            
            # Post-process audio if needed
            if request.speed != 1.0 or request.pitch != 0.0 or request.volume != 0.8:
                await self._post_process_audio(output_file, request)
            
            # Log the duration for debugging
            self.logger.info(f"🎵 Generated {duration:.1f}s of audio from {len(text)} characters")
            
            return str(output_file), duration
            
        except Exception as e:
            self.logger.error(f"Failed to generate audio for chunk: {e}")
            raise
    
    async def _generate_with_coqui(self, text: str, output_file: Path, 
                                 request: AudioGenerationRequest) -> float:
        """Generate audio using Coqui XTTS"""
        
        try:
            # PyTorch 2.6+ compatibility fix - use wrapper script for TTS
            wrapper_script = Path(__file__).parent / "tts_pytorch_fix.py"
            
            # Prepare XTTS command using our PyTorch compatibility wrapper
            cmd = [
                sys.executable, str(wrapper_script),
                "--text", text,
                "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
                "--out_path", str(output_file),
                "--language_idx", "en"
            ]
            
            # Only add speaker reference in full mode (not test mode)
            if not self.test_mode:
                cmd.extend(["--speaker_wav", self._get_speaker_reference(request.voice_model)])
            
            # Run XTTS with PyTorch compatibility fix
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"XTTS failed: {stderr.decode()}")
            
            # Get duration
            if output_file.exists():
                audio_data, sample_rate = librosa.load(str(output_file), sr=None)
                duration = len(audio_data) / sample_rate
                return duration
            else:
                raise RuntimeError("Audio file was not generated")
                
        except Exception as e:
            self.logger.error(f"Coqui TTS generation failed: {e}")
            raise
    
    async def _generate_with_fallback(self, text: str, output_file: Path,
                                    request: AudioGenerationRequest) -> float:
        """Fallback TTS using system tools or alternative engines"""
        
        try:
            # Try Windows SAPI (built-in TTS)
            if self._is_windows():
                duration = await self._generate_with_sapi(text, output_file, request)
            else:
                # Try espeak or festival on Linux/Mac
                duration = await self._generate_with_espeak(text, output_file, request)
            
            return duration
            
        except Exception as e:
            self.logger.error(f"Fallback TTS failed: {e}")
            # Last resort: create silence with text duration estimate
            return self._estimate_speech_duration(text)
    
    async def _generate_with_sapi(self, text: str, output_file: Path,
                                request: AudioGenerationRequest) -> float:
        """Generate audio using Windows SAPI"""
        
        try:
            # Escape text for PowerShell and limit length for test mode
            clean_text = text.replace('"', '""').replace("'", "''")[:500]  # Limit to 500 chars for test mode
            
            # Create temporary PowerShell script file to avoid command line issues
            script_file = self.temp_dir / f"tts_script_{hash(clean_text) % 10000}.ps1"
            
            ps_script_content = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("{str(output_file).replace(chr(92), chr(92) + chr(92))}")
$synth.Rate = {max(-10, min(10, int((request.speed - 1) * 10)))}
$text = @"
{clean_text}
"@
$synth.Speak($text)
$synth.Dispose()
'''
            
            # Write script to file
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(ps_script_content)
            
            # Run PowerShell script file
            process = await asyncio.create_subprocess_exec(
                "powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up script file
            try:
                script_file.unlink()
            except:
                pass
            
            if process.returncode != 0:
                raise RuntimeError(f"PowerShell failed: {stderr.decode()}")
            
            if output_file.exists():
                audio_data, sample_rate = librosa.load(str(output_file), sr=None)
                duration = len(audio_data) / sample_rate
                self.logger.info(f"SAPI TTS generated {duration:.1f}s of audio")
                return duration
            else:
                raise RuntimeError("SAPI TTS failed to generate audio file")
                
        except Exception as e:
            self.logger.error(f"SAPI TTS failed: {e}")
            raise
    
    async def _generate_with_espeak(self, text: str, output_file: Path,
                                  request: AudioGenerationRequest) -> float:
        """Generate audio using espeak"""
        
        try:
            # espeak command
            cmd = [
                "espeak",
                "-s", str(int(150 * request.speed)),  # Speed in words per minute
                "-v", "en",
                "-w", str(output_file),
                text
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if output_file.exists():
                audio_data, sample_rate = librosa.load(str(output_file), sr=None)
                duration = len(audio_data) / sample_rate
                return duration
            else:
                raise RuntimeError("espeak failed to generate audio")
                
        except Exception as e:
            self.logger.error(f"espeak TTS failed: {e}")
            raise
    
    async def _post_process_audio(self, audio_file: Path, request: AudioGenerationRequest):
        """Post-process audio with speed, pitch, and volume adjustments"""
        
        try:
            # Load audio
            audio = AudioSegment.from_wav(str(audio_file))
            
            # Adjust speed
            if request.speed != 1.0:
                # Speed adjustment using frame rate manipulation
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * request.speed)
                })
                audio = audio.set_frame_rate(audio.frame_rate)
            
            # Adjust volume
            if request.volume != 0.8:
                # Convert to dB adjustment
                db_change = 20 * np.log10(request.volume / 0.8)
                audio = audio + db_change
            
            # Export processed audio
            audio.export(str(audio_file), format="wav")
            
        except Exception as e:
            self.logger.warning(f"Audio post-processing failed: {e}")
    
    async def concatenate_audio_segments(self, segments: List[AudioSegmentModel],
                                       output_file: str, add_silence: float = 0.5) -> str:
        """Concatenate multiple audio segments into final audio track"""
        
        try:
            self.logger.info(f"Concatenating {len(segments)} audio segments")
            
            combined_audio = AudioSegment.empty()
            
            for segment in segments:
                if segment.file_path and Path(segment.file_path).exists():
                    # Load segment
                    segment_audio = AudioSegment.from_wav(segment.file_path)
                    
                    # Apply fade in/out to prevent audio pops and clicks
                    if len(segment_audio) > 600:  # Only fade if audio is longer than 0.6 seconds
                        segment_audio = segment_audio.fade_in(300).fade_out(300)  # 300ms fade
                    
                    # Add to combined audio
                    combined_audio += segment_audio
                    
                    # Add silence between segments
                    if add_silence > 0:
                        silence = AudioSegment.silent(duration=int(add_silence * 1000))
                        combined_audio += silence
            
            # Apply final fade in/out to entire audio to ensure smooth start/end
            if len(combined_audio) > 1000:  # Only if audio is longer than 1 second
                combined_audio = combined_audio.fade_in(500).fade_out(500)  # 500ms fade
            
            # Export final audio
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            combined_audio.export(str(output_path), format="wav", 
                                parameters=["-ar", "44100", "-ac", "2"])
            
            self.logger.info(f"Final audio saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Audio concatenation failed: {e}")
            raise
    
    def _clean_script_for_tts(self, text: str) -> str:
        """Clean script text for TTS by removing audio directions, music cues, and speaker labels"""
        
        import re
        
        # Remove structural labels like Part/Chapter and 'Title:' lines
        text = re.sub(r'^(\s*)(Part|Chapter)\s*\d+\s*[:\-].*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^(\s*)Title\s*[:\-].*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove image directives
        text = re.sub(r'^(\s*)Image\s*[:\-].*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^(\s*)Image\s*description\s*[:\-].*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'\[\s*IMAGE\s*:[^\]]*\]', '', text, flags=re.IGNORECASE)

        # Remove music/audio directions in brackets [...]
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove music/audio directions in parentheses (...)
        text = re.sub(r'\([^)]*music[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*sound[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*audio[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*fade[^)]*\)', '', text, flags=re.IGNORECASE)
        
        # Remove speaker labels like "Narrator:", "Host:", etc.
        text = re.sub(r'^[A-Za-z\s]+:\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n[A-Za-z\s]+:\s*', '\n', text)
        
        # Remove stage directions and technical notes
        text = re.sub(r'\*[^*]*\*', '', text)  # *stage directions*
        text = re.sub(r'_[^_]*_', '', text)    # _emphasis notes_
        
        # Remove markdown-style headers / bullets
        text = re.sub(r'^\s*#+\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*]\s+.*$', '', text, flags=re.MULTILINE)
        
        # Collapse excessive spaces, but PRESERVE double newlines as paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize 3+ line breaks down to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace only (preserve quotes in narration)
        text = text.strip()
        
        # Log if text was actually cleaned (this will help debug)
        self.logger.info(f"🧹 Script cleaned for TTS: {len(text)} characters ready for narration")
        
        return text
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Split text into chunks suitable for TTS"""
        
        # First clean the text for TTS
        text = self._clean_script_for_tts(text)
        
        max_size = max_chunk_size or self.chunk_size
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # If paragraph fits in current chunk
            if len(current_chunk + paragraph) < max_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Handle long paragraphs
                if len(paragraph) > max_size:
                    # Split long paragraph by sentences
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if len(temp_chunk + sentence) < max_size:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out empty or very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]
        
        return chunks
    
    def _get_speaker_reference(self, voice_model: str) -> str:
        """Get speaker reference file for voice cloning"""
        
        # For XTTS, we need a reference audio file
        # In a full implementation, you would have collected reference voices
        
        voice_refs = {
            "documentary_narrator": "assets/voices/documentary_narrator.wav",
            "storyteller": "assets/voices/storyteller.wav", 
            "educational": "assets/voices/educational.wav"
        }
        
        ref_file = voice_refs.get(voice_model, voice_refs["documentary_narrator"])
        
        # Create default reference if it doesn't exist
        ref_path = Path(ref_file)
        if not ref_path.exists():
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            # Create a simple tone as placeholder
            self._create_placeholder_reference(ref_path)
        
        return str(ref_path)
    
    def _create_placeholder_reference(self, ref_path: Path):
        """Create a placeholder reference audio file"""
        try:
            # Generate a simple sine wave as placeholder
            sample_rate = 22050
            duration = 3.0  # 3 seconds
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Save as wav
            sf.write(str(ref_path), audio_data, sample_rate)
            
        except Exception as e:
            self.logger.warning(f"Failed to create placeholder reference: {e}")
    
    def _estimate_speech_duration(self, text: str, words_per_minute: int = 150) -> float:
        """Estimate speech duration based on text length"""
        word_count = len(text.split())
        return (word_count / words_per_minute) * 60
    
    def _check_xtts_installation(self) -> bool:
        """Check if Coqui TTS is installed"""
        try:
            import TTS
            return True
        except ImportError:
            return False
    
    async def _install_xtts(self):
        """Install Coqui TTS if not available"""
        self.logger.info("Installing Coqui TTS...")
        
        try:
            process = await asyncio.create_subprocess_exec(
                "pip", "install", "TTS",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("Coqui TTS installed successfully")
            else:
                raise RuntimeError(f"TTS installation failed: {stderr.decode()}")
                
        except Exception as e:
            self.logger.error(f"Failed to install TTS: {e}")
            raise
    
    def _is_coqui_available(self) -> bool:
        """Check if Coqui TTS is available"""
        return self._check_xtts_installation()
    
    def _is_windows(self) -> bool:
        """Check if running on Windows"""
        import platform
        return platform.system() == "Windows"

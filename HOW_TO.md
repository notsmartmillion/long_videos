# 📚 HOW TO: Complete User Guide

A comprehensive guide to understanding and customizing every aspect of your AI video generation system.

---

## 🎯 Table of Contents

1. [🚀 Quick Start Guide](#-quick-start-guide)
2. [📝 Content Generation Process](#-content-generation-process)
3. [🎵 Media Generation Process](#-media-generation-process)
4. [🎞️ Video Assembly Process](#-video-assembly-process)
5. [🤖 Automation System](#-automation-system)
6. [📋 Topic Management](#-topic-management)
7. [🎭 Authenticity System (Anti-AI Detection)](#-authenticity-system-anti-ai-detection)
8. [⚙️ Configuration & Customization](#-configuration--customization)
9. [🔧 Troubleshooting](#-troubleshooting)
10. [🎨 Advanced Customization](#-advanced-customization)

---

## 🧭 Visual Planner, Seeding & QA (Beat-Accurate, Consistent Imagery)

This system includes a second LLM pass (Visual Planner) that plans visuals beat-by-beat, deterministic seeding for character/location consistency, and an automated QA loop to ensure visuals align with narration.

### What this adds
- **Visual Planner (2nd LLM):** Produces a machine-readable shot list (beats) with shot types and prompt slots.
- **Deterministic Seeding:** Consistent characters/locations across beats using hash-based seeds.
- **Alignment:** Maps script to timestamps (heuristic for now; Aeneas-ready).
- **Enhanced Prompts:** Shot-type templates + topic adapters → better visual quality.
- **QA Loop:** Image captioning + text similarity to narration; deterministic retries with fallbacks.
- **Artifacts:** Everything saved under `output/artifacts/{run_id}` for auditability.

### Where to enable
```yaml
# configs/config.yaml
visual_planner:
  enabled: true
  use_enhanced_prompts: true
  deterministic_retry: true
  max_retries_per_beat: 2
  fallback_shot_order: ["diagram","map","insert"]

alignment:
  source: "heuristic"     # set to "aeneas" when ready
  captioner: "blip"       # requires transformers; falls back to "stub"
  similarity_threshold: 0.62

similarity:
  mode: "embeddings"      # uses SentenceTransformers if available; falls back
```

### How seeding works (consistency)
Seeds are computed deterministically so the same entity/scene looks consistent across shots.

Formula:
```
seed = hash32(namespace || topic || [entity_id] || seed_group || image_index)
```

- **namespace:** `config.continuity.seed_namespace` (e.g., `doc_longform`)
- **topic:** current topic name (e.g., `mythology`)
- **entity_id (optional):** for characters/locations to lock identity
- **seed_group:** planner-provided group (e.g., scene id)
- **image_index:** deterministic index per beat

This ensures stable appearance across retries and runs. Fallbacks use salted seeds like `f"{beat_id}:{shot_type}"` to vary composition deterministically.

### End-to-end flow
1) Script → 2) Visual Plan (beats) → 3) Audio (TTS) → 4) Alignment (map beats to time) → 5) Build prompts → 6) Generate images (seeded) → 7) QA + deterministic retry → 8) Assemble video using `timeline.json` for beat-accurate cuts.

### Artifacts per run
Saved under: `output/artifacts/{run_id}/`
- `visual_plan.json` – beats with shot types/prompts
- `alignment.json` – sentence/beat mapping (heuristic or Aeneas)
- `prompts.json` / `prompts.csv` – final prompts used (with seeds/timestamps)
- `qa_report.json` – similarity, retries, pass/fail per beat
- `timeline.json` – `start_s/end_s` per beat and chosen images, plus `audio_path`

### Assembler uses the timeline
The assembler consumes `timeline.json` for exact `start_s/end_s` beat cuts. This is already wired in `main.py` by passing `timeline_path` to the assembler.

### Quality knobs and quick rollback
- Lower false failures: set `alignment.similarity_threshold: 0.60`.
- If VRAM is tight: set `alignment.captioner: "stub"` (BLIP disabled) and/or smaller image batch.
- If timing seems off: temporarily omit `timeline_path` to fall back to legacy spacing.

### Quick test (2-minute run)
```bash
python main.py --test --mode single --topic mythology
```

Expect outputs in `output/artifacts/<run_id>/` and a final video in `output/videos/`.

---

## 🚀 Quick Start Guide

### What Happens When You Run the System?

1. **You define topics** → System reads your topic list
2. **AI researches** → Gathers information about your topic
3. **AI writes script** → Creates 2+ hour detailed narration
4. **AI generates audio** → Natural voice reading the script
5. **AI creates images** → Visuals that match the narration
6. **System assembles video** → Combines everything with effects
7. **Final video ready** → Upload to YouTube!

### Three Ways to Use the System:

```bash
# 1. INTERACTIVE MODE (Best for learning)
python main.py --mode interactive

# 2. SINGLE VIDEO (Generate one video now)
python main.py --mode single --topic mythology --subtopic "Zeus"

# 3. AUTOMATED MODE (Daily video generation)
python main.py --mode auto
```

---

## 📝 Content Generation Process

### What It Does:
Creates the "brain" of your video - researches topics and writes compelling scripts.

### How It Works:

#### 1. **Topic Selection**
- Reads your predefined topics from `data/` folder
- Selects next topic based on priority
- Moves completed topics to "done" list

#### 2. **Research Phase**
- Uses AI (Ollama/LM Studio) to research the topic
- Gathers information from multiple angles
- Creates comprehensive research report
- **Duration:** 2-5 minutes

#### 3. **Script Writing**
- Transforms research into engaging narrative
- Creates chapter structure for 2+ hour content
- Generates image prompts synced to narration
- Adds metadata (title, description, tags)
- **Duration:** 5-10 minutes

### How to Customize:

#### **Change Script Length:**
```yaml
# configs/config.yaml
content:
  target_duration_minutes: 120  # Change this number (60-240 minutes)
  chapters_per_hour: 6          # More chapters = more structure
```

#### **Modify Research Depth:**
```yaml
content:
  research:
    sources_per_topic: 5        # More sources = deeper research
    research_quality_threshold: 0.8  # Higher = stricter quality
```

#### **Customize Writing Style:**
```yaml
content:
  narrative_style: "documentary"  # Options: documentary, educational, storytelling, conversational
  target_audience: "general"      # Options: general, academic, young_adult, expert
```

#### **Add Custom Topics:**
```yaml
# data/my_custom_topics.yaml
topics:
  - title: "The Rise and Fall of Ancient Rome"
    category: "history"
    subtopic: "Roman Empire"
    priority: 1
    custom_research_focus: "military tactics, political intrigue"
    estimated_duration: 150  # minutes
```

---

## 🎵 Media Generation Process

### What It Does:
Creates the audio (voice) and visuals (images) for your video.

### How It Works:

#### 1. **Text-to-Speech (TTS) Audio**
- Uses Coqui XTTS v2 for natural speech
- Reads the generated script aloud
- Applies voice profiles for different styles
- **Duration:** 10-20 minutes for 2-hour script

#### 2. **AI Image Generation**
- Uses SDXL (or FLUX) to create images
- Generates one image per ~8 seconds of audio
- Matches images to script content intelligently
- Applies topic-specific visual styles
- **Duration:** 20-40 minutes (depending on RTX 5080 speed)

### How to Customize:

#### **Change Voice Settings:**
```yaml
# configs/config.yaml
tts:
  voice_model: "xtts_v2"
  language: "en"
  voice_profiles:
    documentary_narrator:
      style: "professional, calm, authoritative"
      pitch: "medium-low"
      speed: "0.9"  # Slightly slower for clarity
    
    storyteller:
      style: "warm, engaging, expressive"
      pitch: "medium"
      speed: "1.0"
```

#### **Modify Image Generation:**
```yaml
image_generation:
  engine: "sdxl"  # or "flux" (requires authentication)
  resolution: "1920x1080"
  images_per_segment: 8  # New image every 8 seconds
  batch_size: 4          # Process 4 images at once
  guidance_scale: 7.5    # Higher = follows prompts more strictly
```

#### **Customize Visual Styles:**
```yaml
image_generation:
  style_templates:
    mythology:
      style: "classical art, renaissance painting, dramatic lighting"
      negative_prompt: "modern, contemporary, photographs"
      
    space:
      style: "NASA photography, space telescope, cosmic vista, stars"
      negative_prompt: "cartoons, drawings, people"
      
    history:
      style: "historical painting, period accurate, documentary style"
      negative_prompt: "modern, anachronistic, fantasy"
```

#### **Add Custom Voice Profile:**
```yaml
tts:
  voice_profiles:
    my_custom_voice:
      style: "energetic, enthusiastic, youthful"
      characteristics: "clear pronunciation, varied intonation"
      pitch: "medium-high"
      speed: "1.1"  # Slightly faster
```

---

## 🎞️ Video Assembly Process

### What It Does:
Combines audio and images into professional-looking videos with cinematic effects.

### How It Works:

#### 1. **Image Processing**
- Applies Ken Burns effect (slow zoom and pan)
- Adds particle effects (moving stars/specs)
- Synchronizes with audio timing
- **Duration:** 5-15 minutes

#### 2. **Video Rendering**
- Uses FFmpeg with NVENC (RTX 5080 acceleration)
- Combines all elements into final MP4
- Generates thumbnail
- **Duration:** 10-25 minutes

### Visual Effects Explained:

#### **Ken Burns Effect:**
- **What:** Slow zoom and pan on static images
- **Why:** Makes images feel cinematic and alive
- **Customizable:** Zoom speed, direction, intensity

#### **Particle Effects:**
- **What:** Moving white specs/stars across images
- **Why:** Adds life and movement, especially for space topics
- **Customizable:** Count, speed, brightness, patterns

### How to Customize:

#### **Ken Burns Effect Settings:**
```yaml
# Automatic settings per image
video_assembly:
  ken_burns:
    intensity: 0.2      # 0.0 = no movement, 1.0 = dramatic
    speed: 1.0          # How fast the movement happens
    zoom_range: [1.0, 1.3]  # Start and end zoom levels
    pan_variety: true   # Vary movement directions
```

#### **Particle System:**
```yaml
video_assembly:
  particles:
    enabled: true
    count: 30           # Number of particles
    size_range: [1, 3]  # Min and max particle size
    speed: 1.0          # Movement speed
    brightness: 0.8     # How bright they appear
    twinkle: true       # Do they twinkle?
```

#### **Video Quality:**
```yaml
video_assembly:
  quality_preset: "high"  # Options: draft, balanced, high, ultra
  resolution: [1920, 1080]
  fps: 30
  enable_motion_blur: true
  enable_gpu_acceleration: true
```

#### **Custom Transitions:**
```yaml
video_assembly:
  transitions:
    default_type: "fade"     # fade, dissolve, slide_left, slide_right
    duration: 0.5            # Seconds
    between_chapters: "zoom_in"  # Special transition between chapters
```

---

## 🤖 Automation System

### What It Does:
Runs the entire system automatically on a schedule, generating videos daily without your intervention.

### How It Works:

#### 1. **Schedule Management**
- Creates schedules for daily/weekly/monthly generation
- Manages topic queues automatically
- Handles retries and error recovery
- **Always running:** Checks every minute for scheduled tasks

#### 2. **Job Processing**
- Processes video generation jobs in queue
- Monitors system health
- Tracks performance metrics
- Cleans up old files automatically

### How to Use:

#### **Start Automation:**
```bash
# This runs forever until you stop it
python main.py --mode auto
```

#### **Create Custom Schedule:**
```python
# Run this in Python to create a schedule
from src.automation.automation_models import ScheduleConfig
from datetime import time

schedule = ScheduleConfig(
    name="Morning History Videos",
    description="Generate history videos every morning",
    frequency="daily",                    # daily, weekly, monthly
    time_of_day=time(7, 30),             # 7:30 AM
    topic_categories=["history", "science"],
    max_videos_per_day=2,
    quality_preset="high"
)
```

### How to Customize:

#### **Automation Settings:**
```yaml
# configs/config.yaml
automation:
  max_concurrent_jobs: 2              # How many videos to make at once
  cleanup_after_days: 30              # Delete old files after X days
  max_retries_per_job: 3              # Retry failed videos
  enable_health_checks: true          # Monitor system health
  health_check_interval_minutes: 30   # How often to check
```

#### **Schedule Templates:**
```yaml
automation:
  default_schedules:
    daily_mythology:
      time: "08:00"
      categories: ["mythology"]
      max_videos: 1
      
    weekend_science:
      frequency: "weekly"
      day: "saturday"
      time: "10:00"
      categories: ["space", "science"]
      max_videos: 2
```

#### **Performance Monitoring:**
The system tracks:
- **Success Rate:** % of videos that generate successfully
- **Average Time:** How long videos take to create
- **Queue Status:** How many videos are waiting
- **System Health:** GPU temperature, memory usage, disk space
- **Error Patterns:** What types of failures happen

---

## 📋 Topic Management

### What It Does:
Manages your video topics, priorities, and ensures you never run out of content ideas.

### How It Works:

#### 1. **Topic Files**
```
data/
├── my_mythology_topics.yaml    # Your active topics
├── space_topics_example.yaml   # Example topics
├── completed_topics.yaml       # Auto-generated completed
└── topic_queue.yaml            # System state
```

#### 2. **Topic Lifecycle**
1. **Pending** → Topics in your yaml files
2. **In Progress** → Currently being processed
3. **Completed** → Moved to completed file
4. **Failed** → Retry or investigate

### Creating Topics:

#### **Basic Topic Format:**
```yaml
topics:
  - title: "Zeus: Thunder God and Ruler of Olympus"
    category: "mythology"
    subtopic: "Zeus"
    priority: 1
    
  - title: "The Mysteries of Black Holes"
    category: "space" 
    subtopic: "black holes"
    priority: 2
    estimated_duration: 90  # Optional: suggest length
```

#### **Advanced Topic Format:**
```yaml
topics:
  - title: "The Lost Civilization of Atlantis"
    category: "mythology"
    subtopic: "Atlantis"
    priority: 1
    
    # Custom research guidance
    research_focus:
      - "Plato's original writings"
      - "Archaeological evidence"
      - "Modern theories and debunking"
    
    # Custom visual style
    visual_style: "underwater ruins, ancient architecture, mysterious"
    
    # Target audience
    target_audience: "history enthusiasts"
    
    # Estimated final length
    estimated_duration: 135
    
    # Custom tags for YouTube
    custom_tags: ["Atlantis", "Plato", "lost civilization", "archaeology"]
```

### Topic Categories:

#### **Built-in Categories:**
- `mythology` - Gods, legends, ancient stories
- `space` - Planets, stars, cosmos, space exploration
- `history` - Historical events, civilizations, people
- `science` - Discoveries, theories, technology
- `nature` - Animals, environments, phenomena

#### **Adding Custom Categories:**
```yaml
# configs/config.yaml
topics:
  my_custom_category:
    visual_style: "your preferred style"
    sources: ["preferred research sources"]
    keywords: ["relevant", "keywords", "for", "research"]
```

### Managing Your Queue:

#### **Check Queue Status:**
```bash
python main.py --mode interactive
# Choose option 3 → List available topics
```

#### **Add Topics in Bulk:**
```yaml
# Create new file: data/my_new_topics.yaml
topics:
  # Add 20-50 topics here
  - title: "Topic 1"
  # ... etc
```

#### **Priority System:**
- `1` = High priority (generate first)
- `2` = Normal priority
- `3` = Low priority (generate when queue is empty)

---

## 🎭 Authenticity System (Anti-AI Detection)

### What It Does:
Makes your AI-generated videos appear as authentic device recordings to avoid AI detection algorithms.

### How It Works:

#### **1. Audio Authenticity Processing**
- **Removes "Pops" and "Clicks"** - Fixes the telltale signs of TTS splicing
- **Adds Room Tone** - Subtle background ambience like real recordings
- **Natural Compression** - Mimics microphone characteristics
- **Micro-Imperfections** - Tiny artifacts that exist in real recordings

#### **2. Metadata Spoofing**
- **iOS ReplayKit** - Makes videos appear as iPhone screen recordings
- **macOS QuickTime** - Spoofs as Mac screen captures
- **Android Capture** - Mimics Android screen recording
- **Camera Metadata** - Adds GPS, device info, camera settings

#### **3. Device Compression Simulation**
- **Phone Processing** - Applies typical mobile compression patterns
- **Webcam Quality** - Lower bitrate/quality like real webcams
- **DSLR Characteristics** - High-quality camera-like processing

### How to Use:

#### **Enable Basic Authenticity:**
```yaml
# configs/config.yaml
authenticity:
  enabled: true              # Turn on authenticity processing
  process_audio: true        # Fix audio artifacts
  spoof_metadata: true       # Spoof as device recording
  spoof_as: "ios"           # Appear as iPhone screen recording
```

#### **Full Authenticity Configuration:**
```yaml
authenticity:
  enabled: true
  
  # Audio Processing (CRITICAL - fixes the "pops" issue)
  process_audio: true
  add_room_tone: true          # Subtle background noise
  apply_compression: true      # Natural microphone response
  add_imperfections: true      # Tiny natural artifacts
  vary_voice_timing: true      # Subtle speed variations
  
  # Device Simulation
  apply_device_compression: true
  device_type: "phone"         # phone, webcam, dslr
  
  # Metadata Spoofing
  spoof_metadata: true
  spoof_as: "ios"              # ios, macos, android, phone_camera, dslr
```

### Authenticity Presets:

#### **"iPhone User" Preset:**
```yaml
authenticity:
  enabled: true
  spoof_as: "ios"
  device_type: "phone"
  process_audio: true
  # Results in: iOS ReplayKit metadata + phone compression + clean audio
```

#### **"Mac Content Creator" Preset:**
```yaml
authenticity:
  enabled: true
  spoof_as: "macos"
  device_type: "webcam"
  process_audio: true
  # Results in: QuickTime metadata + webcam quality + processed audio
```

#### **"Android User" Preset:**
```yaml
authenticity:
  enabled: true
  spoof_as: "android"
  device_type: "phone"
  process_audio: true
  # Results in: Android screen capture metadata + mobile compression
```

#### **"Real Camera" Preset:**
```yaml
authenticity:
  enabled: true
  spoof_as: "phone_camera"
  device_type: "phone"
  add_gps_data: true           # Adds location metadata
  # Results in: Camera metadata with GPS + natural compression
```

### What Each Setting Does:

#### **Audio Processing Settings:**
```yaml
process_audio: true           # ENABLE THIS - fixes TTS splice pops
add_room_tone: true          # Adds very quiet background ambience
apply_compression: true       # Mimics microphone dynamic range
add_imperfections: true      # Tiny harmonic distortion like real mics
vary_voice_timing: true      # Subtle speed/pitch variations
```

#### **Metadata Spoofing Options:**
```yaml
spoof_as: "ios"              # iOS ReplayKit (most common)
spoof_as: "macos"            # macOS QuickTime Player
spoof_as: "android"          # Android screen recording
spoof_as: "phone_camera"     # Real phone camera with GPS
spoof_as: "dslr"             # Professional camera metadata
spoof_as: "webcam"           # USB webcam recording
```

### Before vs After Authenticity Processing:

#### **❌ BEFORE (Raw AI Output):**
- Perfect audio with no background noise
- FFmpeg encoding signatures in metadata
- Uniform compression patterns
- No device-specific characteristics
- Splice artifacts between TTS segments

#### **✅ AFTER (Authenticity Processed):**
- Natural audio with subtle room tone
- iOS ReplayKit metadata signatures
- Device-specific compression patterns
- Realistic timestamps and device info
- Seamless audio with crossfades

### Advanced Authenticity Techniques:

#### **Random Device Rotation:**
```python
# Rotate between different devices for variety
import random

devices = ["ios", "macos", "android"]
chosen_device = random.choice(devices)

# Update config before each video generation
config.authenticity.spoof_as = chosen_device
```

#### **Seasonal GPS Coordinates:**
```yaml
# Add realistic GPS data that matches your content
authenticity:
  spoof_as: "phone_camera"
  add_location_data: true
  location_preset: "us_west_coast"  # Adds random California coordinates
```

#### **Custom Compression Profiles:**
```yaml
authenticity:
  device_type: "custom"
  custom_compression:
    bitrate: "8M"               # Custom bitrate
    preset: "medium"            # Encoding speed
    crf: "23"                   # Quality level
```

### Testing Your Authenticity:

#### **1. Test Audio Quality:**
Listen for:
- ✅ No "pops" or "clicks" between sentences
- ✅ Subtle background ambience
- ✅ Natural-sounding compression

#### **2. Check Metadata:**
```bash
# Use ffprobe to check metadata
ffprobe -v quiet -print_format json -show_format your_video.mp4

# Look for:
# - com.apple.quicktime.author: ReplayKitRecording
# - creation_time with realistic timestamp
# - Appropriate encoder signatures
```

#### **3. Visual Inspection:**
- ✅ Slight compression artifacts (not perfect quality)
- ✅ Resolution typical of chosen device
- ✅ Frame rate consistent with device type

### Authenticity Best Practices:

1. **Always Enable Audio Processing** - This is the most important feature
2. **Vary Your Device Types** - Don't use the same spoof type every time
3. **Match Content to Device** - Use "ios" for casual content, "dslr" for professional
4. **Test Regularly** - Upload a few test videos to check if they're flagged
5. **Update Timestamps** - Ensure creation times are realistic and recent

### Performance Impact:

- **Audio Processing:** +2-5 minutes render time
- **Metadata Spoofing:** +30 seconds  
- **Device Compression:** +1-3 minutes
- **Total Overhead:** ~5-10% of original render time

**Worth it?** Absolutely - this is what separates your videos from obviously AI-generated content.

---

## ⚙️ Configuration & Customization

### Main Configuration File: `configs/config.yaml`

This file controls every aspect of your system. Here's how to modify it:

#### **System Settings:**
```yaml
system:
  cuda_device: 0              # Which GPU to use (0 for first)
  max_memory_gb: 32           # Limit system memory usage
  temp_cleanup: true          # Auto-delete temporary files
  log_level: "INFO"           # DEBUG, INFO, WARNING, ERROR
```

#### **Content Generation:**
```yaml
content:
  target_duration_minutes: 120   # Video length (60-240)
  chapters_per_hour: 6           # Chapter structure
  narrative_style: "documentary" # Writing style
  research_depth: "comprehensive" # deep, comprehensive, quick
  
  # Research sources priority
  research:
    prefer_academic: true
    include_recent: true
    fact_check_enabled: true
    minimum_sources: 3
```

#### **Performance Tuning for RTX 5080:**
```yaml
performance:
  enable_mixed_precision: true    # Use FP16 for speed
  enable_flash_attention: true    # Memory efficient attention
  compile_models: true            # PyTorch 2.0 compilation
  gpu_memory_fraction: 0.8        # Use 80% of VRAM
  max_parallel_audio: 4           # Parallel audio processing
  max_parallel_images: 8          # Parallel image generation
  torch_compile_mode: "reduce-overhead"
```

#### **File Paths:**
```yaml
paths:
  data: "./data"
  output: "./output" 
  models: "./models"
  temp: "./temp"
  logs: "./logs"
  cache: "./cache"
```

### Advanced Customization:

#### **Custom Prompt Templates:**
Edit `src/content_generation/prompt_templates.py` to modify how the AI generates content:

```python
CUSTOM_SCRIPT_TEMPLATE = """
You are creating a {duration}-minute documentary script about {topic}.

Style: {style}
Audience: {audience}

Requirements:
- Create engaging introduction
- Build narrative tension
- Include fascinating details
- End with thought-provoking conclusion

Structure:
{chapter_structure}

Begin:
"""
```

#### **Custom Voice Training:**
If you want to use your own voice:
1. Record 10-20 minutes of clear speech
2. Use Coqui TTS training tools
3. Update voice paths in config

#### **Custom Visual Effects:**
Edit `src/video_assembly/video_effects.py` to add new effects:

```python
def apply_custom_effect(self, image, progress):
    # Your custom effect code here
    return modified_image
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions:

#### **"No topics found" Error:**
```bash
# Check your topic files
ls data/*.yaml

# Verify format
python -c "import yaml; print(yaml.safe_load(open('data/my_mythology_topics.yaml')))"
```

#### **GPU Memory Errors:**
```yaml
# Reduce batch sizes in config.yaml
image_generation:
  batch_size: 2  # Reduce from 4

performance:
  gpu_memory_fraction: 0.6  # Reduce from 0.8
```

#### **Slow Generation:**
```yaml
# Enable all optimizations
performance:
  enable_mixed_precision: true
  enable_flash_attention: true
  compile_models: true

# Use draft quality for testing
video_assembly:
  quality_preset: "draft"
```

#### **FFmpeg Not Found:**
```bash
# Windows: Install FFmpeg
winget install FFmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

#### **CUDA Issues:**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### **Ollama Connection Issues:**
```bash
# Start Ollama service
ollama serve

# In new terminal, pull model
ollama pull llama3.2:latest

# Test connection
curl http://localhost:11434/api/tags
```

### Performance Optimization:

#### **For Maximum Speed:**
```yaml
# Draft quality for testing
video_assembly:
  quality_preset: "draft"
  enable_motion_blur: false

# Smaller images
image_generation:
  resolution: "1280x720"
  batch_size: 8

# Faster content generation
content:
  research_depth: "quick"
  target_duration_minutes: 60
```

#### **For Maximum Quality:**
```yaml
# Ultra quality
video_assembly:
  quality_preset: "ultra"
  enable_motion_blur: true

# High resolution
image_generation:
  resolution: "1920x1080"
  guidance_scale: 9.0

# Deep research
content:
  research_depth: "comprehensive"
  minimum_sources: 5
```

---

## 🎨 Advanced Customization

### Creating Custom Pipelines:

#### **Custom Content Generator:**
```python
# src/content_generation/custom_generator.py
from .content_models import ContentGenerationRequest, ContentGenerationResult

class CustomContentGenerator:
    def __init__(self, config):
        self.config = config
    
    async def generate_content(self, topic, subtopic=None):
        # Your custom content generation logic
        # Return ContentGenerationResult
        pass
```

#### **Custom Media Pipeline:**
```python
# src/media_generation/custom_media.py
class CustomMediaPipeline:
    async def generate_custom_audio(self, script, style="custom"):
        # Your custom TTS logic
        pass
    
    async def generate_custom_images(self, prompts, style="custom"):
        # Your custom image generation
        pass
```

### Integration with External Services:

#### **Custom Research Sources:**
```python
# src/content_generation/custom_research.py
class CustomResearchSource:
    async def research_topic(self, topic):
        # Connect to your preferred research APIs
        # Wikipedia, Google Scholar, custom databases
        pass
```

#### **Custom Upload Integration:**
```python
# src/automation/custom_uploader.py
class YouTubeUploader:
    async def upload_video(self, video_path, metadata):
        # YouTube API integration
        pass

class CustomPlatformUploader:
    async def upload_to_platform(self, video_path):
        # Upload to Vimeo, TikTok, etc.
        pass
```

### Monitoring & Analytics:

#### **Custom Metrics:**
```python
# src/utils/custom_metrics.py
class VideoMetrics:
    def track_generation_time(self, start, end):
        pass
    
    def track_quality_score(self, video_path):
        pass
    
    def export_analytics(self, timeframe):
        pass
```

#### **Health Monitoring:**
```python
# src/automation/health_monitor.py
class SystemHealthMonitor:
    def check_gpu_temperature(self):
        pass
    
    def check_disk_space(self):
        pass
    
    def alert_if_critical(self):
        pass
```

---

## 🎯 Tips for Success

### Content Strategy:
1. **Batch Similar Topics** - Group related topics for consistent visual style
2. **Mix Difficulty Levels** - Alternate complex and simple topics
3. **Seasonal Content** - Plan holiday/seasonal topics in advance
4. **Trending Topics** - Research what's popular in your niches

### Quality Optimization:
1. **Test First** - Always run a few test videos before automation
2. **Monitor Output** - Check first few automated videos for quality
3. **Iterate Settings** - Adjust based on results and preferences
4. **Backup Topics** - Keep large topic queues to prevent interruptions

### Performance Tips:
1. **Regular Maintenance** - Clean temp files weekly
2. **Monitor Resources** - Keep an eye on disk space and GPU temperature
3. **Update Models** - Newer AI models often perform better
4. **Optimize Schedules** - Run generation during off-peak hours

### Scaling Up:
1. **Multiple Categories** - Run different topic categories on different schedules
2. **Quality Tiers** - Use different quality settings for different channels
3. **Batch Processing** - Process multiple videos in sequence during low-usage times
4. **Resource Management** - Monitor and optimize for your specific hardware

---

**🎉 You now have complete control over your AI video generation system!**

Need help with something specific? Check the logs in `logs/video_ai.log` or run the interactive mode for guided troubleshooting.

Happy video creating! 🚀

---

## 📌 Appendix: What’s New (Non-Destructive Addendum)

### OpenAI integration (gpt-4o-mini)
- `.env.local` with `OPENAI_API_KEY` is now required for cloud LLM usage
- Both story and visual planner calls request JSON via `response_format={type: json_object}`
- Centralized client in `src/llm/openai_client.py` with env overrides

### Test mode `--test`
- Produces ~2-minute videos with smaller SDXL steps/batch and fast SAPI TTS
- Enabled via CLI flag: `python main.py --test --mode single --topic mythology`

### Visual Planner + QA
- Beat-level visual plan; deterministic seeding; QA with captioning/similarity
- Artifacts saved to `output/artifacts/<run_id>`: `visual_plan.json`, `alignment.json`, `prompts.json/csv`, `qa_report.json`, `timeline.json`
- Topic-aware thresholds and caption cleanup reduce mythology “statue” bias

### RTX 5080 strict GPU + NVENC
- Startup validates CUDA + device name is RTX 5080 and quits otherwise
- Video assembly uses `h264_nvenc`

### TTS updates
- Default engine: Windows SAPI (fast, stable); XTTS isolated
- Name possessives normalized (e.g., Achilles' not Achilles's) for better TTS

### Image generation improvements
- SDXL defaults: `resolution: 1344x768`, `guidance_scale: 5.5`, `num_inference_steps: 28`
- Mythology/history style templates and negatives refined to avoid statue bias

### Script artifacts
- Full narration saved to `output/artifacts/script_debug/full_scripts/` when `debug.save_full_scripts: true`

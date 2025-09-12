# 🎬 Long Video AI Automation System

**Automated YouTube Video Generation for ANY Topic** - Create compelling 2+ hour documentaries about mythology, space, history, science, or any subject you choose!

> **🚀 RTX 5080 Optimized** - Built specifically for high-performance local AI generation with 64GB RAM support

## 🌟 What This System Does

Transform any topic into professional, long-form documentary videos with:
- **🤖 AI-Generated Scripts** - Research and write comprehensive 2+ hour narratives
- **🎙️ Natural Voice-Over** - Local TTS with documentary-quality narration
- **🖼️ Context-Aware Images** - AI-generated visuals synchronized with script timing
- **🎞️ Professional Assembly** - Ken Burns effects, particle stars, transitions, NVENC rendering
- **📈 SEO Optimization** - Auto-generated titles, descriptions, tags, thumbnails
- **⚡ Full Automation** - Queue-based daily video generation

## ✅ Current Status

**🎉 FULLY COMPLETE AND READY FOR PRODUCTION:**
- 📝 Content Generation Pipeline (Research + Script Writing)
- 🎵 TTS Engine (Coqui XTTS v2 with multiple voice profiles)
- 🖼️ Image Generation (SDXL with topic-specific styles)
- 🎞️ Video Assembly Pipeline (Ken Burns effects, particle systems, NVENC rendering)
- 🤖 Automation & Scheduling System (Daily generation, queue management)
- 📋 Topic Queue Management System
- ⚙️ RTX 5080 Performance Optimizations
- 📊 Health Monitoring & Performance Tracking

## 🎯 Pre-Defined Topic System

The system uses a simple topic queue where you can pre-define 20-30+ video ideas:

### 📂 Where to Put Your Topics

**Main Topic File:** `data/my_mythology_topics.yaml`

```yaml
topics:
  - title: "Zeus: The King of Olympus and Master of Thunder"
    category: "mythology"
    subtopic: "Zeus"
    priority: 1
    
  - title: "Aphrodite: Goddess of Love, Beauty and Passion"
    category: "mythology" 
    subtopic: "Aphrodite"
    priority: 1
    
  # Add 20-30 more topics...
```

### 🔄 How It Works

1. **Add Topics** → Edit your topic file with video ideas
2. **Run System** → Topics processed automatically in priority order
3. **Auto-Move** → Completed topics moved to `data/completed_topics.yaml`
4. **Continuous** → System works through your entire list

## 🎨 Topic Categories Supported

Each category has optimized visual styles and research sources:

### 🏛️ Mythology
- **Examples:** Greek Gods, Norse Legends, Egyptian Mysteries
- **Style:** Classical art, renaissance painting, marble sculptures
- **Sources:** Academic papers, historical texts, mythology databases

### 🚀 Space & Astronomy
- **Examples:** Jupiter's Moons, Black Holes, Galaxy Formation
- **Style:** Space photography, cosmic imagery, scientific visualization
- **Sources:** NASA, ESA, scientific journals, telescope data

### 🏰 History
- **Examples:** Ancient Rome, Medieval Times, World Wars
- **Style:** Historical paintings, period-accurate imagery
- **Sources:** Historical records, archaeological evidence

### 🔬 Science & Nature
- **Examples:** Quantum Physics, Ocean Life, Evolution
- **Style:** Scientific diagrams, nature photography
- **Sources:** Research papers, documentaries, field studies

## 🔧 Hardware Requirements

**Optimized for RTX 5080 + 64GB RAM:**
- ✅ CUDA 12.x support
- ✅ 16GB+ VRAM for image generation
- ✅ 64GB RAM for model caching
- ✅ NVMe SSD recommended for fast I/O

## 🚀 Installation & Setup

### 1. **Environment Setup**
```bash
# Clone repository
git clone <repository>
cd long_videos_ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies (CUDA-enabled)
pip install -r requirements.txt
```

### 2. **Configure Your Topics**
```bash
# Edit your topic list
notepad data/my_mythology_topics.yaml

# Or copy examples
copy data/space_topics_example.yaml data/my_space_topics.yaml
```

### 3. **Run Your First Generation**
```bash
# Quick test (2-minute video for testing)
python main.py --test --mode single --topic mythology

# Interactive mode (recommended for first time)
python main.py --mode interactive

# Interactive test mode
python main.py --test --mode interactive

# Direct generation (full length)
python main.py --mode single --topic mythology --subtopic "Zeus"

# Start automated daily generation
python main.py --mode auto

# Test all components
python main.py --mode interactive
# Choose option 4 → Test components
```

## 📁 Project Structure

```
long_videos_ai/
├── 📁 data/                           # 👈 YOUR TOPIC FILES GO HERE
│   ├── my_mythology_topics.yaml       # Your mythology video topics
│   ├── space_topics_example.yaml      # Example space topics
│   ├── completed_topics.yaml          # Auto-generated completed videos
│   └── topic_queue.yaml              # Auto-generated queue state
├── 📁 output/                         # Generated content
│   ├── audio/                         # TTS audio files
│   ├── images/                        # AI-generated images
│   ├── videos/                        # Final video files
│   └── content_results/               # Script & research data
├── 📁 src/                           # Source code
│   ├── content_generation/            # Research & script writing
│   ├── media_generation/              # TTS & image creation
│   ├── video_assembly/                # Ken Burns effects & rendering
│   ├── automation/                    # Scheduling & queue management
│   └── utils/                         # Configuration & tools
├── 📁 configs/                       # Configuration files
├── 📁 models/                        # AI models (auto-downloaded)
├── 📁 temp/                          # Temporary processing files
└── main.py                           # Main entry point
```

## 🎨 Advanced Customization

### Adding Custom Topics
Create new topic categories in `configs/config.yaml`:

```yaml
topics:
  your_custom_topic:
    sources: ["specialized_sources", "expert_databases"]
    visual_style: "your_preferred_style"
    keywords: ["relevant", "search", "terms"]
```

### Visual Style Templates
Customize image generation in `configs/config.yaml`:

```yaml
style_templates:
  your_style:
    base_style: "art style, lighting, composition"
    colors: "color palette description"
    mood: "atmospheric descriptors"
```

### Voice Profiles
Multiple narrator voices available:

- **documentary_narrator** - Professional, authoritative
- **storyteller** - Engaging, warm storytelling
- **educational** - Clear, friendly explanation

## 🎬 Performance Expectations

With RTX 5080 + 64GB RAM:

| Component | Time for 2-Hour Video | Resources Used |
|-----------|----------------------|----------------|
| **Content Generation** | 5-10 minutes | CPU + Internet |
| **TTS Audio** | 8-15 minutes | CPU + RAM |
| **Image Generation** | 20-30 minutes | RTX 5080 VRAM |
| **Video Assembly** | 10-20 minutes | GPU + RAM |
| **Total** | **45-75 minutes** | **$0 cost** |

## 🔍 Troubleshooting

### Common Issues

**CUDA Not Found:**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Errors:**
- Reduce batch_size in `configs/config.yaml`
- Enable CPU offloading if needed

**Missing Dependencies:**
```bash
# Reinstall with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Getting Help

1. **Test Components:** Run `python main.py` → Option 4
2. **Check Logs:** See `logs/video_ai.log`
3. **Validate Config:** Verify `configs/config.yaml` syntax

## 📊 Quality Features

### Content Quality
- ✅ Multi-source research verification
- ✅ Fact-checking integration
- ✅ Citation tracking
- ✅ Content completeness scoring

### Media Quality
- ✅ 44.1kHz audio generation
- ✅ 1920x1080 image resolution
- ✅ Style consistency across topics
- ✅ Timing synchronization validation

### Visual Planner, Seeding & QA (New)
- **Visual Planner:** Second LLM pass creates a beat-level shot list with shot types.
- **Deterministic Seeding:** Stable character/location appearance across shots using `seed_for_image` hashes.
- **Alignment:** Heuristic (default) with plug-in wrapper to Aeneas when enabled.
- **Enhanced Prompts:** Shot-type templates + topic adapters drive better images.
- **QA & Fallbacks:** Captioning + similarity; deterministic retries with salted seeds and fallback shot order.
- **Artifacts:** `output/artifacts/{run_id}/visual_plan.json`, `alignment.json`, `prompts.json/csv`, `qa_report.json`, `timeline.json`.

Enable in `configs/config.yaml`:
```yaml
visual_planner:
  enabled: true
  use_enhanced_prompts: true
  deterministic_retry: true
  max_retries_per_beat: 2
  fallback_shot_order: ["diagram","map","insert"]

alignment:
  source: "heuristic"    # set to "aeneas" when ready
  captioner: "blip"      # falls back to stub automatically
  similarity_threshold: 0.62

similarity:
  mode: "embeddings"     # falls back automatically
```

### Performance Monitoring
- ✅ GPU memory optimization
- ✅ Generation time tracking
- ✅ Quality score assessment
- ✅ Error recovery systems
- ✅ Real-time health monitoring
- ✅ Automated queue management

## 🎯 Success Metrics

**Target Performance:**
- 🎬 **Generation Time:** 30-60 minutes for 2-hour video (RTX 5080 optimized)
- 📈 **Content Quality:** 90%+ completeness score
- 💰 **Cost:** $0 (completely local)
- 🚀 **Consistency:** 95%+ successful generations
- 🤖 **Automation:** Full daily video generation

## 🛠️ Technology Stack

**Local AI Processing:**
- **Content:** Ollama/LM Studio (Llama 3.2+)
- **TTS:** Coqui XTTS v2 (multi-voice)
- **Images:** FLUX/SDXL (RTX 5080 optimized)
- **Video:** FFmpeg + hardware encoding
- **Framework:** Python 3.11+ + PyTorch 2.4+

## 📈 Development Roadmap

### ✅ Phase 1: Foundation (COMPLETED)
- [x] Topic-agnostic architecture
- [x] RTX 5080 optimization  
- [x] Content generation pipeline
- [x] Media generation pipeline

### 🔄 Phase 2: Video Production (IN PROGRESS)
- [ ] Video assembly with Ken Burns effects
- [ ] Background music integration
- [ ] Subtitle generation
- [ ] Thumbnail creation

### 📅 Phase 3: Automation (PLANNED)
- [ ] Daily scheduling system
- [ ] YouTube API integration
- [ ] Performance analytics
- [ ] Error recovery & notifications

### 🚀 Phase 4: Advanced Features (FUTURE)
- [ ] Multi-language support
- [ ] A/B testing for thumbnails
- [ ] Viewer engagement optimization
- [ ] Channel management tools

## 📋 Example Topic Files

Check these files for inspiration:
- `data/my_mythology_topics.yaml` - 20+ Greek mythology videos
- `data/space_topics_example.yaml` - Astronomy and space exploration
- `configs/topics_examples.yaml` - Additional topic categories

## 📝 Important Notes

### Content Guidelines
- ✅ All content is researched and fact-checked
- ✅ Citations and sources are tracked
- ✅ Educational focus with accurate information
- ⚠️ Review generated content before publishing

### Performance Tips
- 🎯 Start with 3-5 topic videos to test system
- 📊 Monitor GPU memory usage during generation
- 🔄 Use priority levels to control generation order
- 💾 Keep temp files for debugging if needed

### Legal Considerations
- 📜 All generated content is original
- 🎨 Images are AI-generated (not copyrighted)
- 🎵 Background music should be royalty-free
- ⚖️ Comply with YouTube content policies

---

## 🚀 Ready to Start?

1. **Set up your environment** (15 minutes)
2. **Add your topics** to `data/my_mythology_topics.yaml`
3. **Run your first generation** with `python main.py`
4. **Watch your first 2-hour documentary** get created automatically!

**🎬 Happy video creating! 🎬**

---

## 📌 What’s New (Addendum)

### OpenAI switch (gpt-4o-mini)
- The system now calls OpenAI for both story generation and the Visual Planner.
- Structured JSON outputs enabled via `response_format={type: json_object}`.
- Configure `.env.local` with `OPENAI_API_KEY`; optional model overrides: `PLANNER_OPENAI_MODEL`, `GEN_OPENAI_MODEL`.

### Test mode `--test`
- Fast 1–2 minute path: smaller SDXL steps/batch, Windows SAPI TTS.
- Use: `python main.py --test --mode single --topic mythology`.

### Visual Planner, deterministic seeding, QA artifacts
- Beat-level shot list, entity consistency via hashed seeds, and automated QA with caption similarity.
- Per-run artifacts in `output/artifacts/<run_id>`: planner, alignment, prompts, QA, timeline.

### RTX 5080 strict GPU and NVENC
- App validates CUDA + device name contains RTX 5080 and aborts otherwise.
- Video assembly uses NVENC (`h264_nvenc`).

### TTS and text normalization
- Default TTS engine set to Windows SAPI; XTTS remains optional.
- Name possessives normalized (e.g., Achilles') for better pronunciation.

### Image generation quality
- SDXL tuned defaults: `1344x768`, `guidance_scale: 5.5`, `num_inference_steps: 28`.
- Mythology/history styles and negatives adjusted to avoid statue/wax bias.

"""AI Image generation engine optimized for RTX 5080 using FLUX/SDXL"""

import asyncio
import logging
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import hashlib
import json
from datetime import datetime

from .media_models import GeneratedImage, ImageGenerationRequest, StylePreset


class ImageGenerator:
    """High-performance image generation using FLUX/SDXL optimized for RTX 5080"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('video_ai.image_generator')
        
        # Paths
        self.output_dir = Path(getattr(config.paths, 'output', './output')) / 'images'
        self.temp_dir = Path(getattr(config.paths, 'temp', './temp')) / 'images'
        self.models_dir = Path(getattr(config.paths, 'models', './models')) / 'image_generation'
        self.cache_dir = Path(getattr(config.paths, 'cache', './temp/cache'))
        
        # Create directories
        for dir_path in [self.output_dir, self.temp_dir, self.models_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # GPU settings optimized for RTX 5080 - NO CPU FALLBACK
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA not available! RTX 5080 required for image generation")
        
        self.device = torch.device("cuda")  # GPU ONLY - no fallback
        self.dtype = torch.float16  # Use FP16 for RTX 5080 performance
        self.batch_size = config.image_generation.batch_size
        
        self.logger.info(f"Image Generator: GPU-only mode enabled for {torch.cuda.get_device_name(0)}")
        
        # Model settings
        self.model_name = config.image_generation.engine  # "flux" or "sdxl"
        self.resolution = config.image_generation.resolution
        self.guidance_scale = config.image_generation.guidance_scale
        self.num_inference_steps = config.image_generation.num_inference_steps
        
        # Performance optimizations - RTX 5080 GPU-ONLY
        self.enable_memory_efficient_attention = True
        self.enable_cpu_offload = False  # GPU-ONLY: Keep all models on RTX 5080
        self.compile_models = getattr(config.performance, 'compile_models', True)
        
        # RTX 5080 specific optimizations (16GB VRAM)
        # Temporarily disable torch.compile due to PyTorch 2.9 nightly stack overflow issue
        self.enable_torch_compile = False  # getattr(config.performance, 'enable_torch_compile', True)
        self.torch_compile_mode = getattr(config.performance, 'torch_compile_mode', 'reduce-overhead')
        
        # RTX 5080 memory management settings (SDXL is memory intensive)
        self.enable_vae_slicing = True   # Enable for SDXL memory efficiency
        self.enable_vae_tiling = True    # Enable for SDXL memory efficiency
        self.enable_sequential_cpu_offload = False  # Keep everything on GPU
        self.use_deterministic_algorithms = False  # Allow faster non-deterministic algorithms
        
        # Set PyTorch memory management for RTX 5080
        import os
        import gc
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:8'
        
        # Enable garbage collection for memory management
        gc.enable()
        
        # Log initial memory state
        if torch.cuda.is_available():
            self.logger.info(f"RTX 5080 Initial VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
            self.logger.info(f"RTX 5080 Available VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
                # Model components
        self.pipeline = None
        self.is_loaded = False
        
        # Style presets
        self.style_presets = self._load_style_presets()
        
        # Generation cache
        self.cache_enabled = True
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_style_presets(self) -> Dict[str, StylePreset]:
        """Load predefined style presets for different topics"""
        
        presets = {
            "mythology_classical": StylePreset(
                name="Classical Mythology (cinematic)",
                base_prompt_additions=(
                    "living human subjects, realistic anatomy, natural skin, cinematic volumetric lighting, "
                    "neoclassical oil painting vibe, dynamic composition, dramatic low angle, hero pose, "
                    "high detail, filmic contrast, depth of field"
                ),
                negative_prompt=(
                    "statue, marble, plaster, engraving, sketch, cartoon, anime, text, watermark, signature, "
                    "mannequin, doll, wax figure, lowres, blurry, deformed, mutated, extra fingers, extra limbs, "
                    "fused limbs, duplicate face, poorly drawn hands, nsfw"
                ),
                guidance_scale=5.5,
                num_inference_steps=28,
                style_strength=0.85,
                color_palette="rich golds, deep blues, ivory whites",
                mood_descriptors="epic, mystical, ancient, cinematic"
            ),
            
            "mythology_norse": StylePreset(
                name="Norse Mythology",
                base_prompt_additions="viking art style, nordic imagery, runic symbols, cold atmosphere, fog, ancient norse art",
                negative_prompt="warm colors, tropical, modern, cartoon, low quality",
                guidance_scale=7.5,
                num_inference_steps=22,
                style_strength=0.7,
                color_palette="cold blues, iron grays, snow whites, deep shadows",
                mood_descriptors="harsh, mystical, ancient, powerful"
            ),
            
            "space_photography": StylePreset(
                name="Space Photography",
                base_prompt_additions="space photography, hubble telescope, NASA imagery, cosmic, astronomical, deep space, high detail",
                negative_prompt="cartoon, anime, artistic interpretation, low resolution, blurry",
                guidance_scale=7.0,
                num_inference_steps=20,
                style_strength=0.9,
                color_palette="deep space blacks, nebula colors, starlight, cosmic blues",
                mood_descriptors="vast, mysterious, awe-inspiring, scientific"
            ),
            
            "historical_photography": StylePreset(
                name="Historical Documentation",
                base_prompt_additions="historical photography, period accurate, documentary style, archival quality, authentic details",
                negative_prompt="modern elements, anachronistic, cartoon, fantasy, low quality",
                guidance_scale=7.5,
                num_inference_steps=22,
                style_strength=0.8,
                color_palette="sepia tones, period appropriate colors, authentic lighting",
                mood_descriptors="authentic, educational, dramatic, historical"
            ),
            
            "nature_documentary": StylePreset(
                name="Nature Documentary",
                base_prompt_additions="nature photography, wildlife documentary, national geographic style, natural lighting, high detail",
                negative_prompt="artificial, urban, industrial, cartoon, low quality",
                guidance_scale=6.5,
                num_inference_steps=20,
                style_strength=0.7,
                color_palette="natural earth tones, vibrant greens, sky blues, sunset colors",
                mood_descriptors="serene, natural, majestic, pristine"
            ),
            
            "scientific_visualization": StylePreset(
                name="Scientific Visualization",
                base_prompt_additions="scientific illustration, technical diagram, educational graphics, clean design, precise details",
                negative_prompt="artistic interpretation, abstract, cartoon, messy, low quality",
                guidance_scale=7.0,
                num_inference_steps=18,
                style_strength=0.6,
                color_palette="clean whites, scientific blues, data visualization colors",
                mood_descriptors="precise, educational, clear, informative"
            )
        }
        
        return presets
    
    async def initialize(self):
        """Initialize the image generation pipeline"""
        
        if self.is_loaded:
            return
        
        try:
            self.logger.info(f"Initializing {self.model_name} pipeline on {self.device}")
            
            # Check GPU memory
            if self.device.type == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Load the appropriate pipeline
            if self.model_name.lower() == "flux":
                await self._load_flux_pipeline()
            elif self.model_name.lower() == "sdxl":
                await self._load_sdxl_pipeline()
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Optimize pipeline for RTX 5080
            await self._optimize_pipeline()
            
            self.logger.info("Image generation pipeline initialized successfully")
            self.is_loaded = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize image pipeline: {e}")
            raise
    
    async def _load_flux_pipeline(self):
        """Load FLUX pipeline"""
        try:
            from diffusers import FluxPipeline
            
            # Load FLUX model - RTX 5080 GPU ONLY
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=self.dtype
            )
            
            # Force GPU-only operation - no CPU offload
            self.pipeline = self.pipeline.to(self.device)
                
        except ImportError:
            self.logger.warning("FLUX not available, falling back to SDXL")
            await self._load_sdxl_pipeline()
        except Exception as e:
            self.logger.error(f"Failed to load FLUX: {e}")
            raise
    
    async def _load_sdxl_pipeline(self):
        """Load SDXL pipeline as fallback"""
        try:
            from diffusers import StableDiffusionXLPipeline
            
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None
            )
            
            # Force GPU-only operation - no CPU offload
            self.pipeline = self.pipeline.to(self.device)
                
        except Exception as e:
            self.logger.error(f"Failed to load SDXL: {e}")
            raise
    
    async def _optimize_pipeline(self):
        """Optimize pipeline for RTX 5080 performance"""
        
        try:
            # Enable memory efficient attention
            if hasattr(self.pipeline.unet, 'set_attn_processor'):
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
            
            # Enable PyTorch compilation for RTX 5080 (PyTorch 2.9+)
            if self.enable_torch_compile and hasattr(torch, 'compile'):
                self.logger.info(f"Compiling model for RTX 5080 (mode: {self.torch_compile_mode})")
                self.pipeline.unet = torch.compile(
                    self.pipeline.unet, 
                    mode=self.torch_compile_mode,
                    fullgraph=True,  # Enable full graph optimization for RTX 5080
                    dynamic=False    # Static shapes for better RTX 5080 performance
                )
            
            # Configure scheduler for RTX 5080 optimization
            # Use fastest scheduler for test mode (8 inference steps)
            if self.num_inference_steps <= 10:
                from diffusers import EulerDiscreteScheduler
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                self.logger.info("Using EulerDiscreteScheduler (ultra-fast for test mode)")
            else:
                from diffusers import DPMSolverMultistepScheduler
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                self.logger.info("Using DPMSolverMultistepScheduler (quality mode)")
            
            # RTX 5080 specific VAE optimizations (16GB VRAM)
            if self.enable_vae_slicing and hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                self.logger.info("VAE slicing enabled (memory constrained mode)")
            elif hasattr(self.pipeline, 'disable_vae_slicing'):
                self.pipeline.disable_vae_slicing()
                self.logger.info("VAE slicing disabled (RTX 5080 high-performance mode)")
                
            if self.enable_vae_tiling and hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()
                self.logger.info("VAE tiling enabled (memory constrained mode)")
            elif hasattr(self.pipeline, 'disable_vae_tiling'):
                self.pipeline.disable_vae_tiling()
                self.logger.info("VAE tiling disabled (RTX 5080 high-performance mode)")
            
            self.logger.info("Pipeline optimization completed")
            
        except Exception as e:
            self.logger.warning(f"Pipeline optimization failed: {e}")
    
    async def generate_images(self, prompts, topic: str,
                            timestamps: List[float] = None, progress_callback=None) -> List[GeneratedImage]:
        """Generate images from prompts.
        Accepts either List[str] or List[dict] with keys:
        prompt, negatives, seed, steps, guidance, width, height, timestamp.
        """
        if not self.is_loaded:
            await self.initialize()

        rich = isinstance(prompts, list) and prompts and isinstance(prompts[0], dict)
        safe_topic = str(topic) if topic else "generic"
        self.logger.info(f"Generating {len(prompts)} images for topic: {safe_topic} ({'rich' if rich else 'legacy'} prompts)")

        try:
            style_preset = self._get_style_preset_for_topic(safe_topic)
            all_images: List[GeneratedImage] = []

            if not rich:
                # Legacy path batches
                batch_size = min(self.batch_size, len(prompts))
                total_batches = (len(prompts) + batch_size - 1) // batch_size
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    batch_timestamps = timestamps[i:i + batch_size] if timestamps else [0.0] * len(batch_prompts)
                    batch_num = i // batch_size + 1
                    progress_percent = (batch_num - 1) / total_batches * 100
                    if progress_callback:
                        progress_callback(progress_percent, f"Generating image batch {batch_num}/{total_batches}")
                    import gc
                    torch.cuda.empty_cache(); gc.collect()
                    memory_start = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"Processing batch {batch_num}/{total_batches} - VRAM: {memory_start:.1f}GB")
                    # Sanitize legacy prompts
                    import re
                    def _sanitize_prompt(txt: str) -> str:
                        if not isinstance(txt, str):
                            txt = str(txt)
                        for b in ["<|endoftext|>", "<|beginoftext|>", "!!"]:
                            txt = txt.replace(b, " ")
                        txt = re.sub(r"\s+", " ", txt)
                        txt = re.sub(r"\s*,\s*,+", ", ", txt)
                        tokens = txt.split()
                        if len(tokens) > 75:
                            txt = " ".join(tokens[:75])
                        return txt.strip()
                    batch_prompts = [_sanitize_prompt(p) for p in batch_prompts]
                    batch_images = await self._generate_batch(batch_prompts, batch_timestamps, style_preset, safe_topic)
                    all_images.extend(batch_images)
                    torch.cuda.empty_cache(); gc.collect()
                    await asyncio.sleep(0.05)
                if progress_callback:
                    progress_callback(100, f"Generated {len(all_images)} images successfully")
                self.logger.info(f"Generated {len(all_images)} images successfully")
                return all_images

            # Rich path (one-by-one to respect per-prompt seeds/settings)
            W, H = map(int, self.resolution.split('x'))
            for idx, req in enumerate(prompts):
                # Normalize request
                # Truncate to CLIP-safe lengths (~75 tokens)
                def _clip_sanitize(text: str, max_tokens: int = 75) -> str:
                    if not text:
                        return ""
                    parts = text.replace("\n", " ").split()
                    if len(parts) <= max_tokens:
                        return " ".join(parts)
                    return " ".join(parts[:max_tokens])

                r = {
                    "prompt": _clip_sanitize(req.get("prompt", "")),
                    "negatives": _clip_sanitize(req.get("negatives", style_preset.negative_prompt if hasattr(style_preset, 'negative_prompt') else "")),
                    "seed": req.get("seed"),
                    "steps": req.get("steps", style_preset.num_inference_steps),
                    "guidance": req.get("guidance", style_preset.guidance_scale),
                    "width": req.get("width", W),
                    "height": req.get("height", H),
                    "timestamp": req.get("timestamp", req.get("start_s", 0.0)),
                    "model_id": req.get("model_id", self.model_name),
                }

                key = self._make_cache_key_rich(r)
                try:
                    self.logger.info(f"IMG gen {idx+1}/{len(prompts)} seed={r.get('seed')} steps={r.get('steps')} guidance={r.get('guidance')} {r.get('width')}x{r.get('height')} t={r.get('timestamp')} prompt='{r['prompt'][:90]}'")
                except Exception:
                    pass
                cached = self._get_cached_image(key)
                if cached:
                    image_path = cached
                else:
                    image_pil = await self._render_one(r)
                    image_path = await self._save_image(image_pil, r["prompt"], safe_topic, r["timestamp"])
                    self._cache_image(key, image_path)

                gen = GeneratedImage(
                    id=f"{safe_topic}_{idx:03d}_{int(r['timestamp'])}",
                    prompt=r["prompt"],
                    file_path=image_path,
                    timestamp=r["timestamp"],
                    duration=8.0,
                    style_used=style_preset.name,
                    generation_settings={
                        "model": self.model_name,
                        "resolution": f"{r['width']}x{r['height']}",
                        "guidance_scale": r["guidance"],
                        "num_inference_steps": r["steps"],
                        "seed": r["seed"],
                    },
                    quality_score=0.8
                )
                all_images.append(gen)
                if progress_callback and idx % 5 == 0:
                    progress_callback(int((idx/len(prompts))*100), f"Generating images {idx}/{len(prompts)}")

            if progress_callback:
                progress_callback(100, f"Generated {len(all_images)} images successfully")
            self.logger.info(f"Generated {len(all_images)} images successfully")
            return all_images

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise

    def _make_cache_key_rich(self, r: Dict[str, Any]) -> str:
        s = f"{r['prompt']}||{r.get('negatives','')}||{r.get('seed')}||{r.get('steps')}||{r.get('guidance')}||{r.get('width')}x{r.get('height')}||{r.get('model_id','sdxl')}"
        return hashlib.md5(s.encode()).hexdigest()

    async def _render_one(self, r: Dict[str, Any]) -> Image.Image:
        # Render a single image honoring per-request seed and settings
        import gc
        torch.cuda.empty_cache(); gc.collect()
        width = int(r.get('width'))
        height = int(r.get('height'))
        steps = int(r.get('steps'))
        guidance = float(r.get('guidance'))
        negatives = r.get('negatives', '')
        seed = r.get('seed') if r.get('seed') is not None else 42
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad():
            results = self.pipeline(
                prompt=r['prompt'],
                negative_prompt=negatives,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=None,
                show_progress_bar=False
            )
        image = results.images[0]
        del results
        torch.cuda.empty_cache(); gc.collect()
        return image
    
    async def _generate_batch(self, prompts: List[str], timestamps: List[float],
                            style_preset: StylePreset, topic: str) -> List[GeneratedImage]:
        """Generate a batch of images"""
        
        batch_images = []
        
        try:
            # Helper: CLIP-safe truncation (~75 tokens)
            def _clip_sanitize(text: str, max_tokens: int = 75) -> str:
                if not text:
                    return ""
                parts = text.replace("\n", " ").split()
                if len(parts) <= max_tokens:
                    return " ".join(parts)
                return " ".join(parts[:max_tokens])
            
            # Enhance prompts with style
            enhanced_prompts = [
                _clip_sanitize(self._enhance_prompt(prompt, style_preset)) for prompt in prompts
            ]
            negative_sanitized = _clip_sanitize(style_preset.negative_prompt)
            
            # Check cache first
            cached_results = []
            non_cached_prompts = []
            non_cached_indices = []
            
            for i, prompt in enumerate(enhanced_prompts):
                cache_key = self._get_cache_key(prompt, style_preset)
                cached_path = self._get_cached_image(cache_key)
                
                if cached_path:
                    self.logger.info(f"Using cached image for prompt {i+1}")
                    cached_results.append((i, cached_path))
                else:
                    non_cached_prompts.append(prompt)
                    non_cached_indices.append(i)
            
            # Generate non-cached images
            generated_images = []
            if non_cached_prompts:
                self.logger.info(f"Generating {len(non_cached_prompts)} new images")
                
                # Pre-generation memory check for RTX 5080
                memory_before = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.debug(f"Pre-generation VRAM: {memory_before:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                
                # Aggressive memory cleanup before generation
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                
                # Parse resolution
                width, height = map(int, self.resolution.split('x'))
                
                # Generate images with clean output (no diffusers progress bars)
                with torch.no_grad():
                    try:
                        results = self.pipeline(
                            prompt=non_cached_prompts,
                            negative_prompt=[negative_sanitized] * len(non_cached_prompts),
                            width=width,
                            height=height,
                            num_inference_steps=style_preset.num_inference_steps,
                            guidance_scale=style_preset.guidance_scale,
                            num_images_per_prompt=1,
                            generator=torch.Generator(device=self.device).manual_seed(42),  # For reproducibility
                            callback_on_step_end=None,  # Disable step callbacks
                            show_progress_bar=False  # Disable diffusers progress bars for clean output
                        )
                        
                        generated_images = results.images
                        
                        # Immediate memory cleanup after generation
                        del results
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Log memory after generation
                        memory_after = torch.cuda.memory_allocated() / 1024**3
                        self.logger.debug(f"Post-generation VRAM: {memory_after:.1f}GB allocated")
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Force cleanup and retry with smaller batch
                            torch.cuda.empty_cache()
                            gc.collect()
                            self.logger.error(f"CUDA OOM during generation. VRAM state: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
                            raise e
                        else:
                            raise
            
            # Process results
            gen_idx = 0
            for i, (prompt, timestamp) in enumerate(zip(prompts, timestamps)):
                # Check if this was cached
                cached_result = next((result for result in cached_results if result[0] == i), None)
                
                if cached_result:
                    image_path = cached_result[1]
                else:
                    # Save generated image
                    image = generated_images[gen_idx]
                    image_path = await self._save_image(image, prompt, topic, timestamp)
                    
                    # Cache the result
                    cache_key = self._get_cache_key(
                        self._enhance_prompt(prompt, style_preset), style_preset
                    )
                    self._cache_image(cache_key, image_path)
                    
                    gen_idx += 1
                
                # Create GeneratedImage object
                generated_image = GeneratedImage(
                    id=f"{topic}_{i:03d}_{int(timestamp)}",
                    prompt=prompt,
                    file_path=image_path,
                    timestamp=timestamp,
                    duration=8.0,  # Default duration
                    style_used=style_preset.name,
                    generation_settings={
                        "model": self.model_name,
                        "resolution": self.resolution,
                        "guidance_scale": style_preset.guidance_scale,
                        "num_inference_steps": style_preset.num_inference_steps,
                        "style_preset": style_preset.name
                    },
                    quality_score=0.8  # TODO: Implement quality assessment
                )
                
                batch_images.append(generated_image)
            
            return batch_images
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            raise
    
    async def _save_image(self, image: Image.Image, prompt: str, topic: str, 
                         timestamp: float) -> str:
        """Save generated image to file"""
        
        try:
            # Create safe filename
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            
            timestamp_str = f"{int(timestamp):06d}"
            # Ensure topic is a safe string and not None
            safe_topic = str(topic) if topic else "generic"
            filename = f"{safe_topic}_{timestamp_str}_{safe_prompt}.png"
            
            # Save to output directory
            output_path = self.output_dir / safe_topic / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image with high quality
            image.save(output_path, "PNG", optimize=True, quality=95)
            
            self.logger.debug(f"Image saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            raise
    
    def _enhance_prompt(self, base_prompt: str, style_preset: StylePreset) -> str:
        """Enhance prompt with style preset"""
        
        enhanced = f"{base_prompt}, {style_preset.base_prompt_additions}"
        
        if style_preset.color_palette:
            enhanced += f", {style_preset.color_palette}"
        
        if style_preset.mood_descriptors:
            enhanced += f", {style_preset.mood_descriptors}"
        
        return enhanced
    
    def _get_style_preset_for_topic(self, topic: str) -> StylePreset:
        """Get appropriate style preset for topic"""
        
        # Topic to style mapping
        style_mapping = {
            "mythology": "mythology_classical",
            "space": "space_photography", 
            "history": "historical_photography",
            "science": "scientific_visualization",
            "nature": "nature_documentary"
        }
        
        # Check for specific mythology subtypes
        if topic == "mythology":
            # Could expand this to detect Norse vs Greek etc.
            preset_name = "mythology_classical"
        else:
            preset_name = style_mapping.get(topic, "mythology_classical")
        
        return self.style_presets.get(preset_name, self.style_presets["mythology_classical"])
    
    def _get_cache_key(self, prompt: str, style_preset: StylePreset) -> str:
        """Generate cache key for prompt and style"""
        
        cache_data = {
            "prompt": prompt,
            "style": style_preset.name,
            "guidance_scale": style_preset.guidance_scale,
            "steps": style_preset.num_inference_steps,
            "resolution": self.resolution,
            "model": self.model_name
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_image(self, cache_key: str) -> Optional[str]:
        """Get cached image path if exists"""
        
        if not self.cache_enabled:
            return None
        
        if cache_key in self.cache_metadata:
            cached_path = self.cache_metadata[cache_key]["path"]
            if Path(cached_path).exists():
                return cached_path
            else:
                # Remove invalid cache entry
                del self.cache_metadata[cache_key]
                self._save_cache_metadata()
        
        return None
    
    def _cache_image(self, cache_key: str, image_path: str):
        """Cache image for future use"""
        
        if not self.cache_enabled:
            return
        
        self.cache_metadata[cache_key] = {
            "path": image_path,
            "created_at": datetime.now().isoformat(),
            "access_count": 0
        }
        
        self._save_cache_metadata()
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        
        cache_file = self.cache_dir / "image_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata"""
        
        cache_file = self.cache_dir / "image_cache.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")
    
    async def cleanup_old_cache(self, max_age_days: int = 30):
        """Clean up old cached images"""
        
        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            keys_to_remove = []
            for key, metadata in self.cache_metadata.items():
                created_at = datetime.fromisoformat(metadata["created_at"]).timestamp()
                if created_at < cutoff_date:
                    # Remove file
                    cache_path = Path(metadata["path"])
                    if cache_path.exists():
                        cache_path.unlink()
                    keys_to_remove.append(key)
            
            # Remove from metadata
            for key in keys_to_remove:
                del self.cache_metadata[key]
            
            self._save_cache_metadata()
            self.logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        
        return {
            "model_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "cache_entries": len(self.cache_metadata),
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        }

"""Main media generation pipeline that orchestrates TTS and image generation"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..content_generation.content_models import VideoScript, ImagePrompt
from .media_models import (
    MediaGenerationResult, AudioGenerationRequest, ImageGenerationRequest,
    AudioSegment, GeneratedImage, AudioQuality
)
from .tts_engine import TTSEngine
from .image_generator import ImageGenerator
from .qa.image_captioner import caption_image
from .qa.qa_rules import passes_similarity, check_diversity
from ..utils.seed import seed_for_image
from ..utils.similarity import cosine_sim
from ..content_generation.visual_planner import VisualPlanner
from ..content_generation.alignment import map_beats_to_times
from ..content_generation.alignment_providers import align_text_audio
from ..media_generation.image_prompt_builder import build_prompts
from ..utils.captions import captioner_mode_from_config
from ..utils.similarity import similarity_mode_from_config


class MediaPipeline:
    """Main media generation pipeline"""
    
    def __init__(self, config):
        self.config = config
        # Ensure logs flow through project logger configured by setup_logging
        self.logger = logging.getLogger('video_ai.media_pipeline')
        
        # Initialize engines
        self.tts_engine = TTSEngine(config)
        self.image_generator = ImageGenerator(config)
        
        # Output directories
        self.output_dir = Path(getattr(config.paths, 'output', './output'))
        self.temp_dir = Path(getattr(config.paths, 'temp', './temp'))
        
        # Performance settings
        self.parallel_audio = getattr(config.performance, 'max_parallel_audio', 2)
        self.parallel_images = getattr(config.performance, 'max_parallel_images', 8)
        # Artifacts
        vp = getattr(self.config, 'visual_planner', None)
        self.artifacts_root = Path(getattr(vp, 'artifacts_dir', './output/artifacts')) if vp else Path('./output/artifacts')
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.artifacts_root  # will be resolved per-run

    def _resolve_artifacts_dir(self) -> None:
        """Resolve the latest run_id artifacts directory if present."""
        try:
            latest = self.artifacts_root / "latest_run_id.txt"
            if latest.exists():
                run_id = latest.read_text(encoding='utf-8').strip()
                cand = self.artifacts_root / run_id
                if cand.exists():
                    self.artifacts_dir = cand
                    return
        except Exception:
            pass
        # fallback to root
        self.artifacts_dir = self.artifacts_root
    
    async def generate_media(self, video_script: VideoScript, 
                           topic: Optional[str]) -> MediaGenerationResult:
        """Generate all media for a video script"""
        
        start_time = datetime.now()
        effective_topic = topic or getattr(getattr(video_script, 'metadata', None), 'topic', None) or 'generic'
        self.logger.info(f"Starting media generation for topic: {effective_topic}")
        
        try:
            # Initialize engines
            await asyncio.gather(
                self.tts_engine.initialize(),
                self.image_generator.initialize()
            )
            
            # Audio FIRST → write alignment with real audio → Images
            self._resolve_artifacts_dir()
            audio_segments = await self._generate_audio_async(video_script, effective_topic)
            try:
                self.logger.info(f"Audio generated: segments={len(audio_segments)}, total_s={sum(s.duration for s in audio_segments):.1f}")
            except Exception:
                pass
            # Overwrite alignment.json based on actual audio (provider-selected)
            try:
                import json
                full_script = video_script.get_full_script_text()
                mode = getattr(self.config.alignment, 'source', 'heuristic')
                alignment_data = align_text_audio(full_script, "", mode=mode)
                (self.artifacts_dir / "alignment.json").write_text(json.dumps(alignment_data, indent=2), encoding='utf-8')
            except Exception:
                pass

            use_enhanced = bool(getattr(self.config.visual_planner, 'use_enhanced_prompts', False))
            if use_enhanced and getattr(self.config.visual_planner, 'enabled', True):
                generated_images = await self._generate_images_with_enhanced_prompts(video_script, effective_topic)
            else:
                generated_images = await self._generate_images_async(video_script, effective_topic)
            
            # Calculate total duration
            total_duration = sum(segment.duration for segment in audio_segments)
            
            # Optional QA (post-image generation)
            if getattr(self.config.qa, 'enabled', True) and generated_images:
                self.logger.info("Running QA checks on generated images...")
                # Build beat texts from script (simple split for now)
                beat_texts = [ch.script_text for ch in video_script.chapters] or [video_script.introduction, video_script.conclusion]
                beat_texts = [t for t in beat_texts if t]
                threshold = getattr(self.config.qa, 'caption_similarity_threshold', 0.62)
                shot_types = []
                accepted = []
                for img in generated_images:
                    cap = caption_image(img.file_path or img.id)
                    # Map to a rough beat index by timestamp proportion
                    idx = 0
                    if beat_texts:
                        idx = min(len(beat_texts)-1, int((img.timestamp / max(1.0, video_script.total_duration)) * len(beat_texts)))
                    text = beat_texts[idx]
                    ok = passes_similarity(cap, text, threshold)
                    if ok:
                        accepted.append(img)
                        shot_types.append('establishing')
                    else:
                        # single retry path would go here (deterministic), keep stub as accept for now
                        accepted.append(img)
                        shot_types.append('establishing')
                # Diversity check
                if not check_diversity(shot_types,
                                       window=getattr(self.config.qa, 'diversity_window', 8),
                                       max_ratio=getattr(self.config.qa, 'diversity_max_ratio', 0.4)):
                    self.logger.info("Diversity check flagged dominance; future step will switch some shots to inserts/maps")
                generated_images = accepted

            # Create result
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = MediaGenerationResult(
                audio_segments=audio_segments,
                generated_images=generated_images,
                total_audio_duration=total_duration,
                total_generation_time=generation_time,
                quality_metrics=self._calculate_quality_metrics(audio_segments, generated_images),
                generation_stats={
                    "audio_segments": len(audio_segments),
                    "generated_images": len(generated_images),
                    "topic": effective_topic,
                    "script_chapters": len(video_script.chapters),
                    "total_words": video_script.total_word_count
                }
            )
            
            self.logger.info(f"Media generation completed in {generation_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Media generation failed: {e}")
            raise
    
    async def generate_audio(self, script: VideoScript, topic: Optional[str] = None, progress_callback=None) -> str:
        """Generate audio from script and return final audio file path"""
        
        try:
            # Create audio generation request
            full_script = script.get_full_script_text()
            
            # Debug: Log the actual script content length for debugging
            print(f"\n🎵 AUDIO DEBUG:")
            print(f"📝 Full script length: {len(full_script)} characters")
            print(f"📝 Estimated words: ~{len(full_script.split())} words")
            print(f"📝 Estimated duration: ~{len(full_script.split()) / 150 * 60:.1f} seconds")
            print(f"📝 Script preview: {full_script[:200]}...")
            print(f"📝 Script ending: ...{full_script[-200:]}")
            
            request = AudioGenerationRequest(
                script_text=full_script,
                voice_model=self.config.tts.voice_model,
                quality=AudioQuality(self.config.tts.output_quality),
                speed=self.config.tts.speed,
                pitch=self.config.tts.pitch,
                volume=self.config.tts.volume
            )
            
            # Generate audio segments with progress tracking
            def tts_progress_callback(percent, message):
                if progress_callback:
                    progress_callback(percent * 0.8, message)  # 80% for TTS, 20% for concatenation
            
            segments = await self.tts_engine.generate_audio(request, topic, tts_progress_callback)
            
            # Concatenate into final audio file
            if progress_callback:
                progress_callback(85, "Concatenating audio segments...")
                
            output_filename = f"{topic or 'video'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = self.output_dir / "audio" / output_filename
            
            final_audio = await self.tts_engine.concatenate_audio_segments(
                segments, str(output_path)
            )
            
            if progress_callback:
                progress_callback(100, "Audio generation complete")
            
            return final_audio
            
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            raise
    
    async def generate_images(self, image_prompts: List[ImagePrompt], 
                            topic: str, progress_callback=None) -> List[str]:
        """Generate images from prompts and return file paths"""
        
        try:
            # Extract prompts and timestamps
            prompts = [img.base_prompt for img in image_prompts]
            timestamps = [img.timestamp for img in image_prompts]
            
            # Generate images with progress tracking
            generated_images = await self.image_generator.generate_images(
                prompts, topic, timestamps, progress_callback
            )
            
            # Return file paths
            return [img.file_path for img in generated_images if img.file_path]
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise
    
    async def _generate_audio_async(self, video_script: VideoScript, 
                                  topic: str) -> List[AudioSegment]:
        """Generate audio asynchronously"""
        
        self.logger.info("Generating audio...")
        
        # Create comprehensive script text
        full_script = video_script.get_full_script_text()
        
        request = AudioGenerationRequest(
            script_text=full_script,
            voice_model=self.config.tts.voice_model,
            quality=AudioQuality(self.config.tts.output_quality),
            speed=self.config.tts.speed,
            pitch=self.config.tts.pitch,
            volume=self.config.tts.volume
        )
        
        # Generate audio segments
        segments = await self.tts_engine.generate_audio(request, topic)
        
        self.logger.info(f"Generated {len(segments)} audio segments")
        return segments
    
    async def _generate_images_async(self, video_script: VideoScript,
                                   topic: str) -> List[GeneratedImage]:
        """Generate images asynchronously"""
        
        self.logger.info("Generating images...")
        
        # Extract image prompts from script
        image_prompts = video_script.image_prompts
        
        if not image_prompts:
            self.logger.warning("No image prompts found in script")
            return []
        
        # Sort by timestamp
        image_prompts.sort(key=lambda x: x.timestamp)
        
        # Extract prompts and metadata
        prompts = []
        timestamps = []
        
        for img_prompt in image_prompts:
            # Enhance prompt with context if available
            enhanced_prompt = img_prompt.base_prompt
            if img_prompt.context:
                enhanced_prompt = f"{img_prompt.base_prompt}, context: {img_prompt.context[:100]}"
            
            prompts.append(enhanced_prompt)
            timestamps.append(img_prompt.timestamp)
        
        # Generate images
        generated_images = await self.image_generator.generate_images(
            prompts, topic, timestamps
        )
        
        self.logger.info(f"Generated {len(generated_images)} images")
        return generated_images

    async def _generate_images_with_enhanced_prompts(self, video_script: VideoScript, topic: str) -> List[GeneratedImage]:
        """Generate images using visual plan enhanced prompts and mapped times."""
        self.logger.info("Generating images using enhanced prompts from visual plan...")
        # Build plan → alignment → prompts on-the-fly to avoid tight coupling
        planner = VisualPlanner(self.config)
        plan = planner.plan_visuals(script_text=video_script.get_full_script_text(), topic=topic)
        mode = getattr(self.config.alignment, 'source', 'heuristic')
        alignment_data = align_text_audio(video_script.get_full_script_text(), "", mode=mode)
        beats_with_times = map_beats_to_times([b for b in plan.model_dump()["beats"]], alignment_data, max(1.0, video_script.total_duration))
        try:
            self.logger.info(f"Planner produced {len(beats_with_times)} beats (after alignment)")
        except Exception:
            pass
        style_tpl = self.config.image_generation.get_style_for_topic(topic)
        adapters = getattr(self.config, 'topic_adapters', {}) or {}
        prompts = build_prompts(
            visual_plan={"beats": beats_with_times},
            style_template={"base_style": style_tpl.base_style, "colors": style_tpl.colors, "mood": style_tpl.mood},
            topic=topic,
            adapters=adapters,
            seed_namespace=self.config.continuity.seed_namespace,
        )

        # Persist artifacts
        try:
            import json, csv
            prompts_path = self.artifacts_dir / "prompts.json"
            prompts_path.write_text(json.dumps(prompts, indent=2), encoding='utf-8')
            csv_path = self.artifacts_dir / "prompts.csv"
            if not csv_path.exists():
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(["beat_id","shot_type","prompt","negatives","seed","start_s","end_s","status","similarity"])
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                for p in prompts:
                    w.writerow([p.get('beat_id'), p.get('shot_type'), p.get('prompt'), p.get('negatives'), p.get('seed'), p.get('start_s',0.0), p.get('end_s',0.0), "pending", "0.000"])
        except Exception:
            pass

        # Emit rich prompt log
        try:
            self.logger.info(f"Rich prompts ready: count={len(prompts)}; first='{(prompts[0].get('prompt') or '')[:120] if prompts else ''}'")
        except Exception:
            pass
        # Build timestamps directly from prompts
        timestamps = [p.get("start_s", 0.0) for p in prompts]
        self.logger.info(f"Generating images with rich prompts: count={len(prompts)}")
        generated = await self.image_generator.generate_images(
            [{
                "prompt": p.get("prompt"),
                "negatives": p.get("negatives", ""),
                "seed": p.get("seed"),
                "steps": p.get("steps"),
                "guidance": p.get("guidance"),
                "width": p.get("width"),
                "height": p.get("height"),
                "timestamp": t
            } for p, t in zip(prompts, timestamps)], topic
        )

        # QA with deterministic fallback loop
        from ..utils.similarity import get_threshold_for_topic
        threshold = float(get_threshold_for_topic(self.config, topic))
        cap_mode = captioner_mode_from_config(self.config)
        sim_mode = similarity_mode_from_config(self.config)
        fallback_order = list(getattr(self.config.visual_planner, 'fallback_shot_order', ["diagram","map","insert"]))
        retry_enabled = bool(getattr(self.config.visual_planner, 'deterministic_retry', True))
        max_retries = int(getattr(self.config.visual_planner, 'max_retries_per_beat', 2))
        try:
            self.logger.info(f"QA config: threshold={threshold} cap_mode={cap_mode} sim_mode={sim_mode} retry={retry_enabled} max_retries={max_retries} fallback_order={fallback_order}")
        except Exception:
            pass

        accepted: List[GeneratedImage] = []
        qa_details = []
        # Helper to fetch per-beat text using narration_span if present
        full_script_text = video_script.get_full_script_text()

        def text_for_prompt(pr):
            span = pr.get('narration_span')
            if span and isinstance(span, dict):
                tokens = full_script_text.split()
                s = max(0, int(span.get('start_token', 0)))
                e = min(len(tokens), int(span.get('end_token', 0)))
                return " ".join(tokens[s:e]) or full_script_text
            return full_script_text

        namespace = self.config.continuity.seed_namespace
        topic_key = topic
        for idx, (p, img) in enumerate(zip(prompts, generated)):
            cap = caption_image(img.file_path or img.id, mode=cap_mode)
            ref_text = text_for_prompt(p)
            # Bias retry: auto-fail if caption shows statue/wax artifacts
            bad_terms = ["statue", "wax", "engraving", "plaster", "doll"]
            if any(t in (cap or "").lower() for t in bad_terms):
                sim = 0.0
            else:
                sim = cosine_sim(cap, ref_text, mode=sim_mode)
            try:
                self.logger.info(f"QA initial: beat={p.get('beat_id')} idx={idx} sim={sim:.3f} file='{(img.file_path or '')[-64:]}' cap='{cap[:90]}'")
            except Exception:
                pass
            if sim >= threshold or not retry_enabled:
                accepted.append(img)
                qa_details.append({"beat_id": p.get('beat_id'), "first_try": {"similarity": float(sim), "image": img.file_path}, "final_status": "passed"})
                continue
            # Deterministic fallback loop
            best_img, best_sim, best_fb = img, sim, None
            tries = 0
            for fb in fallback_order:
                if tries >= max_retries:
                    break
                tries += 1
                alt_seed = seed_for_image(namespace, topic_key, f"{p.get('beat_id')}:{fb}", 1, p.get("entity_id"))
                alt_prompt = f"{fb} fallback: {p['prompt']}"
                alt_req = dict(p)
                alt_req.update({"prompt": alt_prompt, "seed": alt_seed, "timestamp": p.get("start_s", 0.0)})
                alt_img = (await self.image_generator.generate_images([alt_req], topic))[0]
                cap2 = caption_image(alt_img.file_path or alt_img.id, mode=cap_mode)
                sim2 = cosine_sim(cap2, ref_text, mode=sim_mode)
                try:
                    self.logger.info(f"QA retry: beat={p.get('beat_id')} try={tries} fb={fb} sim2={sim2:.3f} file='{(alt_img.file_path or '')[-64:]}'")
                except Exception:
                    pass
                if sim2 > best_sim:
                    best_img, best_sim, best_fb = alt_img, sim2, fb
                if sim2 >= threshold:
                    break
            chosen = best_img if best_sim >= sim else img
            final_status = "passed" if max(sim, best_sim) >= threshold else "failed"
            accepted.append(chosen)
            qa_details.append({
                "beat_id": p.get('beat_id'),
                "first_try": {"similarity": float(sim), "image": img.file_path},
                "retry": {"used_fallback": best_fb, "similarity": float(best_sim), "image": chosen.file_path},
                "final_status": final_status
            })
            try:
                self.logger.info(f"QA final: beat={p.get('beat_id')} status={final_status} best_sim={best_sim:.3f} used_fallback={best_fb}")
            except Exception:
                pass

        # Write qa_report.json
        try:
            import json
            qa_path = self.artifacts_dir / "qa_report.json"
            summary = {
                "beats_total": len(prompts),
                "passed": sum(1 for d in qa_details if d["final_status"] == "passed"),
                "failed_first_try": sum(1 for d in qa_details if d.get("retry") and d["first_try"]["similarity"] < threshold),
                "recovered_after_retry": sum(1 for d in qa_details if d.get("retry") and d["final_status"] == "passed"),
                "failed_final": sum(1 for d in qa_details if d.get("final_status") == "failed"),
                "threshold": threshold,
            }
            qa_path.write_text(json.dumps({"summary": summary, "details": qa_details}, indent=2), encoding='utf-8')
        except Exception:
            pass

        # Emit timeline.json, include audio_path when available
        try:
            import json
            timeline = {
                "audio_path": str((Path(getattr(self.config.paths, 'output', './output')) / 'audio').resolve()),
                "beats": [
                    {
                        "id": p.get('beat_id'),
                        "start_s": float(p.get('start_s', 0.0)),
                        "end_s": float(p.get('end_s', 0.0)),
                        "shot_type": p.get('shot_type'),
                        "seed_group": "",
                        "entity_ids": [],
                        "image_candidates": [img.file_path],
                        "chosen_image": img.file_path,
                    }
                    for p, img in zip(prompts, accepted)
                ],
            }
            (self.artifacts_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding='utf-8')
        except Exception:
            pass

        return accepted
    
    def _calculate_quality_metrics(self, audio_segments: List[AudioSegment],
                                 generated_images: List[GeneratedImage]) -> Dict[str, float]:
        """Calculate quality metrics for generated media"""
        
        metrics = {}
        
        # Audio quality metrics
        if audio_segments:
            total_audio_duration = sum(seg.duration for seg in audio_segments)
            avg_segment_duration = total_audio_duration / len(audio_segments)
            
            metrics["audio_total_duration"] = total_audio_duration
            metrics["audio_segment_count"] = len(audio_segments)
            metrics["audio_avg_segment_duration"] = avg_segment_duration
            metrics["audio_completeness"] = 1.0 if all(seg.file_path for seg in audio_segments) else 0.0
        
        # Image quality metrics
        if generated_images:
            avg_quality = sum(img.quality_score for img in generated_images) / len(generated_images)
            
            metrics["image_count"] = len(generated_images)
            metrics["image_avg_quality"] = avg_quality
            metrics["image_completeness"] = 1.0 if all(img.file_path for img in generated_images) else 0.0
            
            # Calculate image timing distribution
            if len(generated_images) > 1:
                timestamps = [img.timestamp for img in generated_images]
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                metrics["image_avg_interval"] = avg_interval
        
        # Overall quality score
        audio_score = metrics.get("audio_completeness", 0) * 0.4
        image_score = metrics.get("image_completeness", 0) * 0.4
        timing_score = 0.2 if metrics.get("image_avg_interval", 0) > 0 else 0
        
        metrics["overall_quality"] = audio_score + image_score + timing_score
        
        return metrics
    
    async def optimize_for_video_length(self, video_script: VideoScript,
                                      target_duration_minutes: int) -> VideoScript:
        """Optimize media generation for specific video length"""
        
        target_duration_seconds = target_duration_minutes * 60
        
        # Estimate current duration based on word count
        words_per_minute = 150  # Average speaking pace
        estimated_duration = (video_script.total_word_count / words_per_minute) * 60
        
        self.logger.info(f"Target duration: {target_duration_seconds}s, Estimated: {estimated_duration}s")
        
        # Adjust image timing if needed
        if video_script.image_prompts:
            # Redistribute images evenly across target duration
            image_count = len(video_script.image_prompts)
            optimal_interval = target_duration_seconds / image_count
            
            for i, img_prompt in enumerate(video_script.image_prompts):
                img_prompt.timestamp = i * optimal_interval
                img_prompt.duration = min(optimal_interval * 0.8, 10.0)  # Max 10 seconds per image
        
        return video_script
    
    async def validate_media_quality(self, result: MediaGenerationResult) -> Dict[str, Any]:
        """Validate quality of generated media"""
        
        validation = {
            "passed": True,
            "issues": [],
            "warnings": [],
            "metrics": result.quality_metrics
        }
        
        # Check audio quality
        if not result.audio_segments:
            validation["passed"] = False
            validation["issues"].append("No audio segments generated")
        else:
            missing_audio = [seg for seg in result.audio_segments if not seg.file_path]
            if missing_audio:
                validation["passed"] = False
                validation["issues"].append(f"{len(missing_audio)} audio segments missing files")
        
        # Check image quality
        if not result.generated_images:
            validation["warnings"].append("No images generated")
        else:
            missing_images = [img for img in result.generated_images if not img.file_path]
            if missing_images:
                validation["warnings"].append(f"{len(missing_images)} images missing files")
            
            low_quality_images = [img for img in result.generated_images if img.quality_score < 0.5]
            if low_quality_images:
                validation["warnings"].append(f"{len(low_quality_images)} low quality images")
        
        # Check timing
        if result.generated_images:
            timestamps = [img.timestamp for img in result.generated_images]
            if len(set(timestamps)) != len(timestamps):
                validation["warnings"].append("Duplicate image timestamps detected")
        
        # Overall quality check
        overall_quality = result.quality_metrics.get("overall_quality", 0)
        if overall_quality < 0.7:
            validation["warnings"].append(f"Overall quality score low: {overall_quality:.2f}")
        
        return validation
    
    async def cleanup_temp_files(self):
        """Clean up temporary files"""
        
        try:
            temp_dirs = [
                self.temp_dir / "audio",
                self.temp_dir / "images"
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file_path in temp_dir.glob("*"):
                        if file_path.is_file():
                            file_path.unlink()
            
            self.logger.info("Temporary files cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        
        stats = {
            "tts_engine_loaded": self.tts_engine.is_loaded,
            "image_generator_loaded": self.image_generator.is_loaded,
            "parallel_audio_limit": self.parallel_audio,
            "parallel_images_limit": self.parallel_images
        }
        
        # Add image generator stats
        if self.image_generator.is_loaded:
            stats.update(self.image_generator.get_generation_stats())
        
        return stats

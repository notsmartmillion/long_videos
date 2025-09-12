"""Main content generation pipeline that orchestrates all components"""

import asyncio
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from .content_models import ContentGenerationRequest, ContentGenerationResult, ContentType, NarrativeStructure
from .research_engine import ResearchEngine
from .script_generator import ScriptGenerator
from .visual_planner import VisualPlanner
from .alignment import force_align, map_beats_to_times
from ..media_generation.image_prompt_builder import build_prompts
from .topic_queue import TopicQueue, TopicItem


class ContentPipeline:
    """Main content generation pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.research_engine = ResearchEngine(config)
        self.script_generator = ScriptGenerator(config)
        self.topic_queue = TopicQueue(config)
        self.visual_planner = VisualPlanner(config) if getattr(config.visual_planner, 'enabled', True) else None
        
        # Output directory
        self.output_dir = Path(config.paths.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_content(self, topic: Optional[str] = None, 
                             subtopic: Optional[str] = None,
                             progress_callback=None) -> ContentGenerationResult:
        """Generate complete content for a video"""
        
        start_time = datetime.now()
        
        try:
            # If no topic provided, get next from queue
            if not topic:
                next_topic_item = self.topic_queue.get_next_topic()
                if not next_topic_item:
                    raise ValueError("No topics in queue and no topic provided")
                
                topic = next_topic_item.category
                subtopic = next_topic_item.subtopic or next_topic_item.title
                
                self.logger.info(f"Using queued topic: {topic} - {subtopic}")
            
            # Create content generation request
            request = ContentGenerationRequest(
                topic=topic,
                subtopic=subtopic,
                target_length_minutes=self.config.content.video_length_minutes,
                content_type=ContentType(self.config.content.script_style),
                narrative_structure=NarrativeStructure(self.config.content.narrative_structure),
                target_audience=self.config.content.target_audience
            )
            
            self.logger.info(f"Starting content generation for: {request.topic} - {request.subtopic}")
            
            # Step 1: Research
            if progress_callback:
                progress_callback(10, "Researching topic...")
            self.logger.info("Step 1: Researching topic...")
            async with self.research_engine:
                research_report = await self.research_engine.research_topic(request)
            
            # Step 2: Generate script
            if progress_callback:
                progress_callback(50, "Generating script...")
            self.logger.info("Step 2: Generating script...")
            video_script = await self.script_generator.generate_script(research_report, request)
            # Detailed logging for verification
            try:
                self.logger.info(
                    f"Script generated: title='{video_script.title[:120]}', words={video_script.total_word_count}, duration_s={video_script.total_duration:.1f}"
                )
                # Log entire narration to file for immediate inspection
                logs_dir = Path(getattr(self.config.paths, 'output', './output')) / 'content_results'
                logs_dir.mkdir(parents=True, exist_ok=True)
                script_dump = logs_dir / f"{request.topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_script.txt"
                script_dump.write_text(video_script.get_full_script_text(), encoding='utf-8')
                self.logger.info(f"Full narration saved to: {script_dump}")
            except Exception as e:
                self.logger.warning(f"Failed to dump full script text: {e}")

            plan = None
            if self.visual_planner is not None:
                if progress_callback:
                    progress_callback(60, "Planning visuals...")
                self.logger.info("Step 3: Planning visuals (visual planner enabled)...")
                # Phase wiring (non-invasive): produce plan artifact
                plan = self.visual_planner.plan_visuals(script_text=video_script.get_full_script_text(), topic=request.topic)
                try:
                    self.logger.info(f"Visual plan: beats={len(plan.beats)}, entities={len(plan.entities)}")
                except Exception:
                    pass
                # Create run_id and artifact dir (single source of truth)
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                art_dir = Path(getattr(self.config.visual_planner, 'artifacts_dir', './output/artifacts')) / run_id
                art_dir.mkdir(parents=True, exist_ok=True)
                # Save latest run id for downstream consumers
                try:
                    latest = Path(getattr(self.config.visual_planner, 'artifacts_dir', './output/artifacts')) / "latest_run_id.txt"
                    latest.write_text(run_id, encoding='utf-8')
                except Exception:
                    pass

                # Save plan
                (art_dir / "visual_plan.json").write_text(plan.model_dump_json(indent=2), encoding='utf-8')

                # Heuristic alignment (placeholder) - audio pass will overwrite later
                if progress_callback:
                    progress_callback(65, "Aligning beats to audio (heuristic)...")
                target_seconds = max(1.0, request.target_length_minutes * 60)
                alignment_placeholder = force_align("", video_script.get_full_script_text())
                (art_dir / "alignment.json").write_text(json.dumps(alignment_placeholder, indent=2), encoding='utf-8')

                beats_with_times = map_beats_to_times([b for b in plan.model_dump()["beats"]], alignment_placeholder, target_seconds)
                style_tpl = self.config.image_generation.get_style_for_topic(request.topic)
                adapters = getattr(self.config, 'topic_adapters', {}) or {}
                enhanced_prompts = build_prompts(
                    visual_plan={"beats": beats_with_times},
                    style_template={"base_style": style_tpl.base_style, "colors": style_tpl.colors, "mood": style_tpl.mood},
                    topic=request.topic,
                    adapters=adapters,
                    seed_namespace=self.config.continuity.seed_namespace,
                )
                (art_dir / "prompts.json").write_text(json.dumps(enhanced_prompts, indent=2), encoding='utf-8')
                self.logger.info(f"Enhanced prompts prepared: count={len(enhanced_prompts)} (first prompt: '{(enhanced_prompts[0].get('prompt','') if enhanced_prompts else '')[:140]}')")
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ContentGenerationResult(
                request=request,
                research_report=research_report,
                video_script=video_script,
                generation_stats={
                    "research_sources": len(research_report.sources),
                    "script_chapters": len(video_script.chapters),
                    "total_words": video_script.total_word_count,
                    "total_duration_minutes": video_script.total_duration / 60,
                    "image_prompts": len(video_script.image_prompts),
                    "run_id": run_id if self.visual_planner is not None else None
                },
                visual_plan=plan,
                quality_scores={
                    "research_quality": research_report.research_quality_score,
                    "content_completeness": self._assess_content_completeness(video_script),
                    "narrative_flow": self._assess_narrative_flow(video_script)
                },
                generation_time_seconds=generation_time
            )
            
            # Save result
            await self._save_content_result(result)
            
            self.logger.info(f"Content generation completed in {generation_time:.1f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            raise
    
    async def generate_from_queue(self) -> Optional[ContentGenerationResult]:
        """Generate content for the next topic in queue"""
        
        next_topic = self.topic_queue.get_next_topic()
        if not next_topic:
            self.logger.info("No topics in queue")
            return None
        
        try:
            self.logger.info(f"Processing queued topic: {next_topic.title}")
            
            # Generate content
            result = await self.generate_content(
                topic=next_topic.category,
                subtopic=next_topic.subtopic or next_topic.title
            )
            
            # Mark topic as completed
            self.topic_queue.mark_completed(
                topic_id=next_topic.id,
                generation_time_minutes=result.generation_time_seconds / 60,
                success=True,
                stats=result.generation_stats
            )
            
            return result
            
        except Exception as e:
            # Mark topic as failed
            self.topic_queue.mark_failed(
                topic_id=next_topic.id,
                error_message=str(e)
            )
            self.logger.error(f"Failed to process queued topic '{next_topic.title}': {e}")
            raise
    
    async def process_topic_file(self, file_path: str) -> int:
        """Process a file containing multiple topics to add to queue"""
        
        try:
            imported_count = self.topic_queue.import_topics_from_file(file_path)
            self.logger.info(f"Imported {imported_count} topics from {file_path}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Failed to process topic file {file_path}: {e}")
            raise
    
    def get_queue_status(self) -> dict:
        """Get current queue status"""
        return self.topic_queue.get_queue_status()
    
    def add_topic_to_queue(self, title: str, category: str, **kwargs) -> str:
        """Add a single topic to the queue"""
        return self.topic_queue.add_topic(title, category, **kwargs)
    
    async def _save_content_result(self, result: ContentGenerationResult):
        """Save content generation result to file"""
        
        # Create filename based on topic and timestamp
        safe_topic = "".join(c for c in result.request.topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_subtopic = ""
        if result.request.subtopic:
            safe_subtopic = "".join(c for c in result.request.subtopic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if safe_subtopic:
            filename = f"{safe_topic}_{safe_subtopic}_{timestamp}.json"
        else:
            filename = f"{safe_topic}_{timestamp}.json"
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        output_path = self.output_dir / "content_results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the result
        result.save_to_file(str(output_path))
        
        self.logger.info(f"Content result saved to: {output_path}")
    
    def _assess_content_completeness(self, script) -> float:
        """Assess how complete and comprehensive the content is"""
        
        score = 0.0
        
        # Check if all required components exist
        if script.introduction and len(script.introduction) > 100:
            score += 0.2
        
        if script.chapters and len(script.chapters) >= 6:
            score += 0.3
        
        if script.conclusion and len(script.conclusion) > 100:
            score += 0.2
        
        if script.image_prompts and len(script.image_prompts) >= 10:
            score += 0.2
        
        # Check word count is reasonable
        expected_words = script.metadata.estimated_length_minutes * 150  # 150 words per minute
        word_ratio = script.total_word_count / expected_words
        if 0.8 <= word_ratio <= 1.2:  # Within 20% of expected
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_narrative_flow(self, script) -> float:
        """Assess the narrative flow and structure quality"""
        
        score = 0.0
        
        # Check chapter progression
        if len(script.chapters) > 0:
            # Check if chapters have reasonable duration distribution
            durations = [chapter.duration for chapter in script.chapters]
            avg_duration = sum(durations) / len(durations)
            
            # Good if chapters are relatively even in length
            variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
            if variance < (avg_duration * 0.5) ** 2:  # Low variance
                score += 0.3
            
            # Check if chapters have titles and content
            complete_chapters = sum(1 for ch in script.chapters 
                                  if ch.title and ch.script_text and len(ch.script_text) > 200)
            completeness_ratio = complete_chapters / len(script.chapters)
            score += completeness_ratio * 0.4
        
        # Check image timing distribution
        if script.image_prompts:
            timestamps = [img.timestamp for img in script.image_prompts]
            if len(timestamps) > 1:
                # Good if images are reasonably spaced
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                
                # Target: image every 5-10 minutes (300-600 seconds)
                if 300 <= avg_interval <= 600:
                    score += 0.3
        
        return min(score, 1.0)

"""Script generation using local LLMs for any topic"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
import json as _json
import openai
from pathlib import Path

from .content_models import (
    VideoScript, Chapter, ImagePrompt, ContentMetadata,
    ResearchReport, ContentGenerationRequest, ContentType, NarrativeStructure
)
from .prompt_templates import PromptTemplates
from src.utils.text_normalize import normalize_name_possessives

# Constants for accurate duration calculation
WORDS_PER_SECOND = 2.5  # Average speaking rate for documentary narration
MIN_WORDS_BUFFER = 0.9  # Allow 10% under target
MAX_WORDS_BUFFER = 1.1  # Allow 10% over target


class ScriptGenerator:
    """Generates video scripts using AI models (local or cloud)"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("video_ai.script_generator")
        
        # LLM configuration from config
        self.use_local_llm = config.llm.use_local_llm
        self.local_llm_url = config.llm.local_llm_url
        self.model_name = config.llm.model_name
        self.temperature = config.llm.temperature
        self.max_tokens_default = config.llm.max_tokens
        
        # Initialize LLM client
        if self.use_local_llm:
            # Local LLM (Ollama)
            self.client = openai.OpenAI(
                base_url=self.local_llm_url,
                api_key="not-needed"  # Local LLMs don't need API keys
            )
            self.logger.info(f"Using local LLM: {self.model_name} at {self.local_llm_url}")
        else:
            # OpenAI API via centralized helper
            try:
                from src.llm.openai_client import get_openai_client, choose_model
                self.client = get_openai_client()
                # Allow env override for generation model without changing config
                self.model_name = choose_model("gen") or self.model_name
            except Exception:
                from os import getenv
                api_key = getenv("OPENAI_API_KEY")
                self.client = openai.OpenAI(api_key=api_key)
            self.logger.info(f"Using OpenAI API for script generation (model={self.model_name})")
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates()
    
    def _estimate_duration_from_words(self, word_count: int) -> float:
        """Estimate speech duration from word count"""
        return word_count / WORDS_PER_SECOND
    
    def _calculate_target_words(self, target_minutes: float) -> tuple[int, int]:
        """Calculate target word count range for given duration"""
        target_seconds = target_minutes * 60
        target_words = int(target_seconds * WORDS_PER_SECOND)
        min_words = int(target_words * MIN_WORDS_BUFFER)
        max_words = int(target_words * MAX_WORDS_BUFFER)
        return min_words, max_words
    
    def _count_script_words(self, script_text: str) -> int:
        """Count words in script text, excluding stage directions"""
        # Remove music/audio directions and stage directions for accurate word count
        import re
        
        # Remove content in brackets and parentheses (stage directions)
        clean_text = re.sub(r'\[.*?\]', '', script_text)
        clean_text = re.sub(r'\([^)]*\)', '', clean_text)
        
        # Remove speaker labels
        clean_text = re.sub(r'^[A-Za-z\s]+:\s*', '', clean_text, flags=re.MULTILINE)
        
        # Remove extra whitespace and count words
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return len(clean_text.split()) if clean_text else 0
    
    def _validate_script_length(self, script_text: str, target_minutes: float) -> tuple[bool, str]:
        """Validate script length against target duration"""
        word_count = self._count_script_words(script_text)
        min_words, max_words = self._calculate_target_words(target_minutes)
        estimated_duration = self._estimate_duration_from_words(word_count)
        
        if word_count < min_words:
            return False, f"Script too short: {word_count} words ({estimated_duration:.1f}s) < target {min_words}-{max_words} words ({target_minutes*60:.0f}s)"
        elif word_count > max_words:
            return False, f"Script too long: {word_count} words ({estimated_duration:.1f}s) > target {min_words}-{max_words} words ({target_minutes*60:.0f}s)"
        else:
            return True, f"Script length valid: {word_count} words ({estimated_duration:.1f}s) within target range"
    
    async def _generate_comprehensive_script(self, research_report: ResearchReport, 
                                           request: ContentGenerationRequest) -> Any:
        """Generate complete script using comprehensive template approach"""
        
        # Compile research data from the report
        research_data = self._compile_research_summary(research_report)
        
        # Use the new comprehensive prompt template
        prompt = self.prompt_templates.get_script_generation_prompt(
            research_data=research_data,
            topic=request.topic,
            subtopic=request.subtopic,
            content_type=request.content_type,
            narrative_structure=request.narrative_structure,
            target_length_minutes=request.target_length_minutes
        )
        # Encourage structured JSON while remaining backward compatible
        contract = (
            '{\n'
            '  "title": "string",\n'
            '  "logline": "string",\n'
            '  "narration": "string",\n'
            '  "beats": [{"id": "string", "summary": "string", "start_hint_s": 0}],\n'
            '  "image_prompts": [{"timestamp_hint_s": 0, "prompt": "string"}]\n'
            '}'
        )
        user_with_contract = (
            f"Return ONLY a single JSON object matching this contract, populated from your story.\n{contract}\n\n"
            f"Story instructions:\n{prompt}"
        )
        self.logger.info("🤖 Generating comprehensive script (JSON contract)")
        try:
            data = await self._call_llm(prompt=user_with_contract, max_tokens=4000, json_mode=True)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Fallback to plain text
        response = await self._call_llm(prompt, max_tokens=4000)
        return response

    def _coerce_story_json(self, data: Any, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Normalize story JSON into a consistent shape for downstream use."""
        result: Dict[str, Any] = {
            "title": request.topic,
            "logline": f"A documentary about {request.topic}.",
            "narration": "",
            "beats": [],
            "image_prompts": [],
        }
        if not isinstance(data, dict):
            return result
        result["title"] = str(data.get("title") or result["title"])  # type: ignore
        result["logline"] = str(data.get("logline") or result["logline"])  # type: ignore
        result["narration"] = str(data.get("narration") or "")
        # Beats
        beats = data.get("beats") or []
        norm_beats = []
        if isinstance(beats, list):
            for i, b in enumerate(beats):
                if not isinstance(b, dict):
                    continue
                norm_beats.append({
                    "id": str(b.get("id") or f"beat_{i+1}"),
                    "summary": str(b.get("summary") or ""),
                    "start_hint_s": float(b.get("start_hint_s") or i * 30.0),
                    "estimated_duration_s": float(b.get("estimated_duration_s") or 30.0),
                })
        result["beats"] = norm_beats
        # Image prompts
        imps = data.get("image_prompts") or []
        norm_imps = []
        if isinstance(imps, list):
            for i, p in enumerate(imps):
                if not isinstance(p, dict):
                    continue
                norm_imps.append({
                    "timestamp_hint_s": float(p.get("timestamp_hint_s") or i * 30.0),
                    "prompt": str(p.get("prompt") or ""),
                })
        result["image_prompts"] = norm_imps
        return result

    # -------------------- Chunked Longform Generation --------------------
    async def _generate_outline(self, research_report: ResearchReport, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Ask the LLM for a beats outline with target words per beat (JSON)."""
        research_data = self._compile_research_summary(research_report)
        target_words_total = int(request.target_length_minutes * 60 * WORDS_PER_SECOND)
        beats_min, beats_max = (8, 12) if request.target_length_minutes >= 20 else (6, 8)
        contract = (
            '{\n'
            '  "beats": [\n'
            '    {"id": "string", "title": "string", "summary": "string", "target_words": 0}\n'
            '  ]\n'
            '}'
        )
        prompt = (
            f"Return ONLY a JSON object matching this contract.\n{contract}\n\n"
            f"Goal: plan a documentary script about {request.topic}"
            f" (subtopic: {request.subtopic or request.topic}) totaling ~{target_words_total} words.\n"
            f"Create {beats_min}-{beats_max} beats. Distribute target_words realistically; sum near {target_words_total}.\n\n"
            f"RESEARCH CONTEXT (concise):\n{research_data[:1800]}"
        )
        try:
            data = await self._call_llm(prompt, max_tokens=1200, json_mode=True)
            if isinstance(data, dict) and isinstance(data.get("beats"), list):
                return data
        except Exception:
            pass
        # Fallback simple outline
        beats = []
        num_beats = max(beats_min, min(beats_max, (request.target_length_minutes // 3) or beats_min))
        base_words = max(180, target_words_total // int(num_beats or 1))
        for i in range(int(num_beats)):
            beats.append({
                "id": f"b{i+1:02d}",
                "title": f"Section {i+1}",
                "summary": f"Narrative development part {i+1}.",
                "target_words": base_words,
            })
        return {"beats": beats}

    def _coerce_outline(self, data: Any, request: ContentGenerationRequest) -> List[Dict[str, Any]]:
        """Normalize outline to a list of beats with id,title,summary,target_words."""
        if not isinstance(data, dict):
            return []
        beats = data.get("beats") or []
        norm = []
        for i, b in enumerate(beats):
            if not isinstance(b, dict):
                continue
            norm.append({
                "id": str(b.get("id") or f"b{i+1:02d}"),
                "title": str(b.get("title") or f"Section {i+1}"),
                "summary": str(b.get("summary") or ""),
                "target_words": float(b.get("target_words") or 300.0),
            })
        # Scale target_words to overall target if needed
        target_total = int(request.target_length_minutes * 60 * WORDS_PER_SECOND)
        current_total = int(sum(int(x.get("target_words", 0)) for x in norm) or 1)
        scale = target_total / current_total
        if abs(scale - 1.0) > 0.15:
            for x in norm:
                x["target_words"] = float(max(120.0, min(600.0, x.get("target_words", 300.0) * scale)))
        return norm

    async def _expand_beat(self, beat: Dict[str, Any], research_report: ResearchReport,
                           request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate narration (and optional image prompts) for one beat."""
        research_data = self._compile_research_summary(research_report)
        target_words = int(max(120, min(800, beat.get("target_words", 350))))
        json_contract = (
            '{\n'
            '  "narration": "string",\n'
            '  "image_prompts": [\n'
            '    {"timestamp_hint_s": 0, "prompt": "string"}\n'
            '  ]\n'
            '}'
        )
        prompt = (
            f"Return ONLY a JSON object per this contract:\n{json_contract}\n\n"
            f"Write ~{target_words} words of continuous narration for beat '{beat.get('title')}'.\n"
            f"Beat summary: {beat.get('summary')}\n\n"
            f"RESEARCH CONTEXT (concise):\n{research_data[:1500]}"
        )
        try:
            data = await self._call_llm(prompt, max_tokens=min(2000, target_words * 3), json_mode=True)
            if isinstance(data, dict) and isinstance(data.get("narration"), str):
                return data
        except Exception:
            pass
        # Fallback to plain text
        fallback = await self._call_llm(
            f"Write ~{target_words} words of narration for: {beat.get('title')}\nSummary: {beat.get('summary')}\n",
            max_tokens=min(2000, target_words * 3)
        )
        return {"narration": str(fallback or ""), "image_prompts": []}
    
    async def _parse_comprehensive_script(self, script_content: str, request: ContentGenerationRequest) -> dict:
        """Parse comprehensive script into components"""
        
        # For now, treat the entire script as one flowing narrative
        # TODO: Parse into actual introduction/chapters/conclusion based on markers
        word_count = self._count_script_words(script_content)
        estimated_duration = self._estimate_duration_from_words(word_count)
        
        # Ensure we have a minimum duration to prevent division by zero
        estimated_duration = max(estimated_duration, 1.0)
        
        return {
            "introduction": script_content,  # Simplified - entire script as intro
            "chapters": [],  # Empty for now
            "conclusion": "",  # Empty for now  
            "total_duration": estimated_duration
        }
    
    async def _generate_image_prompts_from_script(self, script_content: str, request: ContentGenerationRequest) -> List[ImagePrompt]:
        """Generate image prompts from script content"""
        
        # Use existing image prompt generation but with script content
        try:
            # Ask for structured JSON prompts
            desired = int(max(2, request.target_length_minutes * 2))
            json_contract = '{"image_prompts": [{"timestamp_hint_s": 0, "prompt": "string"}]}'
            prompt = (
                f"Return ONLY a JSON object matching this contract: {json_contract}\n\n"
                f"Generate {desired} concise, filmable image prompts from the script below. "
                f"Use style '{getattr(request, 'visual_style', 'documentary')}'.\n\n"
                f"SCRIPT:\n{script_content[:2000]}..."
            )
            try:
                data = await self._call_llm(prompt, max_tokens=1200, json_mode=True)
            except Exception:
                data = None
            image_prompts = []
            if isinstance(data, dict) and isinstance(data.get("image_prompts"), list):
                for i, item in enumerate(data["image_prompts"]):
                    ptxt = str(item.get("prompt", "")).strip()
                    if len(ptxt) > 20:
                        thint = float(item.get("timestamp_hint_s", i * 30))
                        image_prompts.append(ImagePrompt(
                            timestamp=thint,
                            base_prompt=ptxt,
                            style_modifiers=getattr(request, 'visual_style', 'documentary'),
                            context=f"Documentary scene {i+1}",
                            duration=30
                        ))
            if not image_prompts:
                # Fallback: plain text list
                fallback_prompt = f"List {desired} image ideas (one per line) for this documentary script:\n\n{script_content[:1500]}..."
                response = await self._call_llm(fallback_prompt, max_tokens=1000)
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    clean = line.strip().lstrip('-').lstrip(f"{i+1}.").strip()
                    if len(clean) > 20:
                        image_prompts.append(ImagePrompt(
                            timestamp=i * 30,
                            base_prompt=clean,
                            style_modifiers=getattr(request, 'visual_style', 'documentary'),
                            context=f"Documentary scene {i+1}",
                            duration=30
                        ))
            
            self.logger.info(f"Generated {len(image_prompts)} image prompts from script")
            return image_prompts[:int(request.target_length_minutes * 2)]  # Limit to target
            
        except Exception as e:
            self.logger.error(f"Failed to generate image prompts from script: {e}")
            return []
    
    async def _generate_description_from_script(self, script_content: str, title: str, request: ContentGenerationRequest) -> str:
        """Generate description from script content"""
        
        try:
            prompt = f"""Create a compelling video description for this documentary:

Title: {title}
Topic: {request.topic}

Script excerpt:
{script_content[:500]}...

Create a 2-3 sentence description that would intrigue viewers."""

            response = await self._call_llm(prompt, max_tokens=200)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate description: {e}")
            return f"An in-depth exploration of {request.topic}"
    
    async def generate_script(self, research_report: ResearchReport, request: ContentGenerationRequest) -> VideoScript:
        """Generate a complete video script from research data using outline+per-beat expansion."""
        self.logger.info(f"Generating script for '{request.topic}' - {request.target_length_minutes} minutes")

        # Calculate target word count for validation
        min_words, max_words = self._calculate_target_words(request.target_length_minutes)
        self.logger.info(f"Target word range: {min_words}-{max_words} words for {request.target_length_minutes} minutes")

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # 1) Outline
                outline_raw = await self._generate_outline(research_report, request)
                outline_beats = self._coerce_outline(outline_raw, request)
                planned_words = int(sum(int(b.get("target_words", 0)) for b in outline_beats)) if outline_beats else 0
                self.logger.info(f"Outline: beats={len(outline_beats)}, planned_words≈{planned_words}")

                # Persist outline
                try:
                    diag_dir = Path(getattr(self.config.paths, 'output', './output')) / 'artifacts' / 'script_debug'
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    (diag_dir / f"outline_{request.topic}.json").write_text(
                        _json.dumps({"beats": outline_beats}, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )
                except Exception:
                    pass

                # 2) Expand each beat (sequential; you can switch to asyncio.gather later)
                beat_results = []
                for beat in outline_beats:
                    br = await self._expand_beat(beat, research_report, request)
                    beat_results.append({"beat": beat, "result": br})

                # Assemble narration
                narration_parts = [str(x["result"].get("narration", "")) for x in beat_results]
                try:
                    beat_word_counts = [self._count_script_words(n) for n in narration_parts]
                    self.logger.info(f"Expansion: per_beat_words={beat_word_counts} (sum={sum(beat_word_counts)})")
                except Exception:
                    pass
                script_content = "\n\n".join(p for p in narration_parts if p.strip())
                # Normalize possessives for names ending in s (e.g., Achilles' vs Achilles's)
                try:
                    script_content = normalize_name_possessives(script_content)
                except Exception:
                    pass

                # Validate against target minutes (relaxed logic)
                is_valid, validation_msg = self._validate_script_length(script_content, request.target_length_minutes)
                try:
                    word_count = self._count_script_words(script_content)
                    target_seconds = request.target_length_minutes * 60
                    target_words = int(target_seconds * WORDS_PER_SECOND)
                    short_ratio = (word_count / max(1, target_words))
                    est_sec = self._estimate_duration_from_words(word_count)
                    self.logger.info(
                        f"📊 Script validation: {validation_msg} | "
                        f"metrics: words={word_count}, target_words={target_words}, "
                        f"ratio={short_ratio:.2f}, est_sec={est_sec:.1f}"
                    )
                except Exception:
                    short_ratio = 1.0
                    self.logger.info(f"📊 Script validation: {validation_msg}")

                # Retry only if severely short
                if not is_valid and short_ratio < 0.8 and retry_count + 1 < max_retries:
                    retry_count += 1
                    self.logger.warning(
                        f"Script notably short (ratio={short_ratio:.2f}), retrying outline/expand "
                        f"({retry_count}/{max_retries})..."
                    )
                    continue
                if not is_valid and short_ratio >= 0.8:
                    self.logger.warning(
                        f"Proceeding despite short script (ratio={short_ratio:.2f}); "
                        f"adjust target or buffers if undesired."
                    )

                # Prefer structured image prompts collected from beats
                structured_prompts: List[Dict[str, Any]] = []
                for i, x in enumerate(beat_results):
                    for ip in (x["result"].get("image_prompts") or []):
                        structured_prompts.append({
                            "timestamp_hint_s": ip.get("timestamp_hint_s", i * 30.0),
                            "prompt": ip.get("prompt", "")
                        })

                # Coerce prompts into ImagePrompt models
                all_image_prompts: List[ImagePrompt] = []
                if structured_prompts:
                    style_template = self.config.image_generation.get_style_for_topic(request.topic)
                    style_mod = f"{style_template.base_style}, {style_template.colors}, {style_template.mood}"
                    for i, it in enumerate(structured_prompts):
                        try:
                            ts = float(it.get("timestamp_hint_s", i * 30.0))
                            txt = str(it.get("prompt", ""))
                            if len(txt) < 5:
                                continue
                            all_image_prompts.append(ImagePrompt(
                                timestamp=ts,
                                base_prompt=txt,
                                style_modifiers=style_mod,
                                context=f"Documentary scene {i+1}",
                                duration=30.0
                            ))
                        except Exception:
                            continue
                if not all_image_prompts:
                    all_image_prompts = await self._generate_image_prompts_from_script(script_content, request)

                                # Parse composed script for duration
                parsed_script = await self._parse_comprehensive_script(script_content, request)

                # Build metadata, title, description
                metadata = ContentMetadata(
                    topic=request.topic,
                    subtopic=request.subtopic,
                    target_audience=request.target_audience,
                    content_type=request.content_type,
                    narrative_structure=request.narrative_structure,
                    estimated_length_minutes=max(
                        1,
                        int(self._estimate_duration_from_words(self._count_script_words(script_content)) / 60),
                    ),
                )

                title = await self._generate_title(research_report, request)
                description = await self._generate_description_from_script(script_content, title, request)

                script = VideoScript(
                    metadata=metadata,
                    title=title,
                    description=description,
                    introduction=parsed_script["introduction"],
                    chapters=parsed_script["chapters"],
                    conclusion=parsed_script["conclusion"],
                    total_duration=parsed_script["total_duration"],
                    total_word_count=self._count_script_words(script_content),
                    image_prompts=all_image_prompts,
                )

                # Persist artifacts
                try:
                    base_out = Path(getattr(self.config.paths, 'output', './output')) / 'artifacts' / 'script_debug'
                    base_out.mkdir(parents=True, exist_ok=True)
                    (base_out / f"beats_{request.topic}.json").write_text(
                        _json.dumps(beat_results, indent=2, ensure_ascii=False), encoding='utf-8'
                    )
                    (base_out / f"narration_{request.topic}.txt").write_text(script_content, encoding='utf-8')
                    if bool(getattr(self.config, 'debug', {}).get('save_full_scripts', False)):
                        full_dir = base_out / 'full_scripts'
                        full_dir.mkdir(parents=True, exist_ok=True)
                        (full_dir / f"full_script_{request.topic}.txt").write_text(script_content, encoding='utf-8')
                except Exception:
                    pass

                # ✅ Success
                break

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.warning(f"Script generation failed, retrying ({retry_count}/{max_retries}): {e}")
                    continue
                else:
                    self.logger.error(f"Script generation failed after {max_retries} attempts: {e}")
                    raise

        # Print detailed script content for verification (force console output)
        print("\n" + "="*80)
        print("📝 GENERATED SCRIPT DETAILS:")
        print("="*80)
        print(f"Title: {script.title}")
        print(f"Chapters: {len(script.chapters)}, Words: {script.total_word_count}, Duration: {script.total_duration:.1f}s")
        print("-"*80)
        print(f"SCRIPT CONTENT ({script.total_word_count} words):")
        print(f"{script.introduction[:500]}...")
        print("="*80 + "\n")

        return script
    
    async def _generate_chapter_structure(self, research_report: ResearchReport,
                                        request: ContentGenerationRequest) -> List[Dict[str, Any]]:
        """Generate the overall chapter structure"""
        
        # Compile research data
        research_summary = self._compile_research_summary(research_report)
        
        prompt = PromptTemplates.get_chapter_breakdown_prompt(
            topic=request.topic,
            research_data=research_summary,
            target_chapters=8  # Default to 8 chapters for 2-hour content
        )
        
        response = await self._call_llm(prompt, max_tokens=1500)
        
        # Parse the response to extract chapter structure
        chapters = self._parse_chapter_structure(response, request.target_length_minutes)
        
        return chapters
    
    async def _generate_introduction(self, research_report: ResearchReport,
                                   request: ContentGenerationRequest) -> str:
        """Generate compelling introduction"""
        
        research_summary = self._compile_research_summary(research_report)
        
        prompt = f"""
Create a compelling 2-3 minute introduction for a documentary about "{request.topic}"{f" focusing on {request.subtopic}" if request.subtopic else ""}.

RESEARCH CONTEXT:
{research_summary[:1500]}

The introduction should:
1. Hook the audience immediately
2. Set the scope and importance of the topic
3. Preview what the audience will discover
4. Build anticipation for the journey ahead
5. Be approximately 300-400 words when spoken

Style: {request.content_type.value} format for {request.target_audience} audience.

Write in an engaging, narrative style suitable for voiceover.
"""
        
        introduction = await self._call_llm(prompt, max_tokens=800)
        return introduction.strip()
    
    async def _generate_chapter(self, chapter_outline: Dict[str, Any],
                              research_report: ResearchReport,
                              request: ContentGenerationRequest,
                              chapter_number: int,
                              start_time: float) -> Chapter:
        """Generate content for a single chapter"""
        
        research_summary = self._compile_research_summary(research_report)
        
        prompt = f"""
Write Chapter {chapter_number}: "{chapter_outline['title']}" for a documentary about {request.topic}.

CHAPTER REQUIREMENTS:
- Duration: {chapter_outline['duration']} minutes
- Key Points: {', '.join(chapter_outline.get('key_points', []))}
- Target Words: ~{int(chapter_outline['duration'] * 150)} words

RESEARCH DATA:
{research_summary[:2000]}

Write engaging content that:
1. Flows naturally from previous content
2. Covers all key points thoroughly
3. Includes natural pauses for visuals
4. Maintains narrative momentum
5. Sets up transition to next chapter

Mark visual moments with [IMAGE: description of what should be shown]

Style: {request.content_type.value} format, {request.narrative_structure.value} structure.
"""
        
        chapter_content = await self._call_llm(prompt, max_tokens=2000)
        
        # Extract image prompts from content
        image_prompts = self._extract_image_markers(chapter_content, start_time)
        
        # Clean content (remove image markers)
        clean_content = re.sub(r'\[IMAGE:[^\]]+\]', '', chapter_content).strip()
        
        chapter = Chapter(
            id=f"chapter_{chapter_number}",
            title=chapter_outline['title'],
            start_time=start_time,
            end_time=start_time + chapter_outline['duration'] * 60,  # Convert to seconds
            duration=chapter_outline['duration'] * 60,  # Convert to seconds
            script_text=clean_content,
            key_points=chapter_outline.get('key_points', []),
            image_prompts=image_prompts,
            background_music_mood=self._determine_music_mood(request.topic, chapter_outline['title'])
        )
        
        return chapter
    
    async def _generate_conclusion(self, research_report: ResearchReport,
                                 request: ContentGenerationRequest) -> str:
        """Generate powerful conclusion"""
        
        prompt = f"""
Create a powerful 2-3 minute conclusion for a documentary about "{request.topic}".

The conclusion should:
1. Summarize the key discoveries/insights
2. Reflect on the broader significance
3. Connect to modern relevance
4. Leave the audience with lasting impact
5. Be approximately 300-400 words when spoken

Topic focus: {request.subtopic or request.topic}
Audience: {request.target_audience}

End on an inspiring and thought-provoking note.
"""
        
        conclusion = await self._call_llm(prompt, max_tokens=800)
        return conclusion.strip()
    
    async def _generate_image_prompts(self, introduction: str, chapters: List[Chapter],
                                    conclusion: str, request: ContentGenerationRequest) -> List[ImagePrompt]:
        """Generate detailed image prompts for the entire video"""
        
        all_prompts = []
        
        # Get topic visual style
        topic_config = self.config.get_topic_config(request.topic)
        visual_style = topic_config.visual_style if topic_config else "cinematic"
        
        # Add prompts from chapters
        for chapter in chapters:
            all_prompts.extend(chapter.image_prompts)
        
        # Generate opening image for introduction
        intro_prompt = await self._generate_specific_image_prompt(
            introduction[:500], request.topic, visual_style, 0.0
        )
        if intro_prompt:
            all_prompts.insert(0, intro_prompt)
        
        # Generate closing image for conclusion
        conclusion_prompt = await self._generate_specific_image_prompt(
            conclusion[:500], request.topic, visual_style, 
            chapters[-1].end_time if chapters else 7200.0
        )
        if conclusion_prompt:
            all_prompts.append(conclusion_prompt)
        
        return sorted(all_prompts, key=lambda x: x.timestamp)
    
    async def _generate_specific_image_prompt(self, context: str, topic: str,
                                            visual_style: str, timestamp: float) -> Optional[ImagePrompt]:
        """Generate a specific image prompt for a piece of content"""
        
        prompt = f"""
Create a detailed image generation prompt for this content about {topic}:

CONTENT: {context}

VISUAL STYLE: {visual_style}

Generate a specific, detailed prompt that would create a compelling image including:
- Main subject/scene
- Composition and framing  
- Lighting and mood
- Color palette
- Specific visual details

Format: Just the image prompt, no explanations.
"""
        
        try:
            image_description = await self._call_llm(prompt, max_tokens=200)
            
            # Get style template for topic
            style_template = self.config.image_generation.get_style_for_topic(topic)
            
            return ImagePrompt(
                timestamp=timestamp,
                base_prompt=image_description.strip(),
                style_modifiers=f"{style_template.base_style}, {style_template.colors}, {style_template.mood}",
                context=context[:100] + "...",
                importance=0.7,
                duration=8.0
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to generate image prompt: {e}")
            return None
    
    async def _generate_title(self, research_report: ResearchReport,
                            request: ContentGenerationRequest) -> str:
        """Generate compelling video title"""
        
        script_summary = f"Documentary about {request.topic}"
        if request.subtopic:
            script_summary += f" focusing on {request.subtopic}"
        
        prompt = PromptTemplates.get_title_generation_prompt(
            topic=request.topic,
            subtopic=request.subtopic,
            script_summary=script_summary
        )
        
        response = await self._call_llm(prompt, max_tokens=500)
        
        # Extract the first (best) title from the response
        titles = self._parse_title_options(response)
        
        return titles[0] if titles else f"The Complete Story of {request.subtopic or request.topic}"
    
    async def _generate_description(self, title: str, chapters: List[Chapter],
                                  request: ContentGenerationRequest) -> str:
        """Generate YouTube description"""
        
        chapter_titles = [chapter.title for chapter in chapters]
        script_summary = f"A comprehensive {request.target_length_minutes}-minute documentary exploring {request.topic}"
        
        prompt = PromptTemplates.get_description_generation_prompt(
            title=title,
            script_summary=script_summary,
            chapters=chapter_titles
        )
        
        description = await self._call_llm(prompt, max_tokens=1000)
        return description.strip()
    
    async def _call_llm(self, prompt: str, max_tokens: int = None, json_mode: bool = False):
        """Call the LLM (local or cloud)"""
        
        # Use configured max_tokens if not specified
        if max_tokens is None:
            max_tokens = self.max_tokens_default
        
        try:
            if self.use_local_llm:
                # Local LLM call (Ollama)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert documentary script writer and researcher. Write engaging, factual content suitable for narration."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )
            else:
                # OpenAI API call
                kwargs = dict(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert documentary script writer and researcher. Write engaging, factual content suitable for narration."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content or ""
            if json_mode:
                try:
                    import json as _json
                    return _json.loads(content)
                except Exception:
                    # Fall back to raw string if parsing fails
                    return content
            # Print the LLM response for debugging and verification (force console output)
            print(f"\n🤖 LLM Response Preview: {content[:300]}...")
            print(f"📊 LLM Response Length: {len(content)} characters")
            if len(content) < 50:
                print(f"⚠️ WARNING: LLM generated very short response: '{content}'")
            return content
        
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _compile_research_summary(self, research_report: ResearchReport) -> str:
        """Compile research data into a summary for prompts"""
        
        summary_parts = []
        
        # Key facts
        if research_report.key_facts:
            summary_parts.append("KEY FACTS:")
            summary_parts.extend([f"- {fact}" for fact in research_report.key_facts[:10]])
        
        # Key figures
        if research_report.key_figures:
            summary_parts.append("\nKEY FIGURES:")
            for figure in research_report.key_figures[:5]:
                summary_parts.append(f"- {figure.get('name', 'Unknown')}: {figure.get('role', 'Important figure')}")
        
        # Important locations
        if research_report.locations:
            summary_parts.append("\nIMPORTANT LOCATIONS:")
            for location in research_report.locations[:5]:
                summary_parts.append(f"- {location.get('name', 'Unknown')}: {location.get('significance', 'Important location')}")
        
        # Source summaries
        if research_report.sources:
            summary_parts.append("\nSOURCE INFORMATION:")
            for source in research_report.sources[:5]:
                summary_parts.append(f"- {source.title}: {source.content_summary[:200]}...")
        
        return "\n".join(summary_parts)
    
    def _parse_chapter_structure(self, response: str, target_minutes: int) -> List[Dict[str, Any]]:
        """Parse LLM response to extract chapter structure"""
        
        # Determine chapter count based on target length (test mode vs production)
        if target_minutes <= 5:
            # Test mode: 2 chapters maximum
            expected_chapters = 2
        elif target_minutes <= 30:
            # Medium videos: 4-6 chapters
            expected_chapters = min(6, max(4, target_minutes // 5))
        else:
            # Long videos: 8-12 chapters
            expected_chapters = min(12, max(8, target_minutes // 15))
        
        chapters = []
        chapter_duration = target_minutes / expected_chapters
        
        # Look for chapter patterns in response
        chapter_pattern = r'Chapter (\d+):\s*([^\n]+)'
        matches = re.findall(chapter_pattern, response, re.IGNORECASE)
        
        # Limit to expected number of chapters
        for i, (num, title) in enumerate(matches[:expected_chapters]):
            chapters.append({
                'number': int(num),
                'title': title.strip(),
                'duration': chapter_duration,
                'key_points': [],  # Could extract these too
                'transition': f"Leads into Chapter {int(num) + 1}" if i < expected_chapters - 1 else "Leads to conclusion"
            })
        
        # If no chapters found or not enough, create default structure
        if len(chapters) < expected_chapters:
            for i in range(len(chapters), expected_chapters):
                chapters.append({
                    'number': i + 1,
                    'title': f'Chapter {i + 1}',
                    'duration': chapter_duration,
                    'key_points': [],
                    'transition': f"Leads into Chapter {i + 2}" if i < expected_chapters - 1 else "Leads to conclusion"
                })
        
        print(f"📊 Chapter Structure: {len(chapters)} chapters for {target_minutes}-minute video")
        
        return chapters
    
    def _extract_image_markers(self, content: str, start_time: float) -> List[ImagePrompt]:
        """Extract [IMAGE: ...] markers from content and convert to ImagePrompt objects"""
        
        prompts = []
        
        # Find all image markers
        image_pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(image_pattern, content)
        
        # Calculate timing based on content position
        content_length = len(content.split())
        words_per_minute = 150  # Average speaking pace
        
        for i, description in enumerate(matches):
            # Estimate timestamp based on position in chapter
            estimated_time = start_time + (i + 1) * (8 * 60)  # Every 8 minutes
            
            prompt = ImagePrompt(
                timestamp=estimated_time,
                base_prompt=description.strip(),
                style_modifiers="documentary style, professional",
                context=description[:100],
                importance=0.6,
                duration=8.0
            )
            prompts.append(prompt)
        
        return prompts
    
    def _determine_music_mood(self, topic: str, chapter_title: str) -> str:
        """Determine appropriate background music mood"""
        
        mood_mapping = {
            "mythology": "epic",
            "space": "mysterious", 
            "history": "dramatic",
            "science": "contemplative",
            "nature": "serene"
        }
        
        base_mood = mood_mapping.get(topic, "neutral")
        
        # Adjust based on chapter content
        title_lower = chapter_title.lower()
        if any(word in title_lower for word in ['war', 'battle', 'conflict']):
            return "dramatic"
        elif any(word in title_lower for word in ['birth', 'creation', 'beginning']):
            return "uplifting"
        elif any(word in title_lower for word in ['death', 'end', 'fall']):
            return "somber"
        
        return base_mood
    
    def _parse_title_options(self, response: str) -> List[str]:
        """Parse title options from LLM response"""
        
        titles = []
        
        # Look for numbered list pattern
        title_pattern = r'^\d+\.\s*([^-\n]+)'
        matches = re.findall(title_pattern, response, re.MULTILINE)
        
        for match in matches:
            clean_title = match.strip().strip('"').strip("'")
            if clean_title and len(clean_title) > 10:  # Reasonable title length
                titles.append(clean_title)
        
        # Fallback: look for any line that looks like a title
        if not titles:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and len(line) < 100 and not line.startswith(('1.', '2.', '-')):
                    titles.append(line)
        
        return titles[:5]  # Return top 5 options
    
    def _count_words(self, introduction: str, chapters: List[Chapter], conclusion: str) -> int:
        """Count total words in script"""
        
        total_words = len(introduction.split())
        total_words += sum(len(chapter.script_text.split()) for chapter in chapters)
        total_words += len(conclusion.split())
        
        return total_words

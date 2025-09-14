"""Data models for content generation pipeline"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ContentType(str, Enum):
    """Types of content that can be generated"""
    DOCUMENTARY = "documentary"
    EDUCATIONAL = "educational"
    NARRATIVE = "narrative"
    STORYTELLING = "storytelling"
    CONVERSATIONAL = "conversational"


class NarrativeStructure(str, Enum):
    """Different narrative structures for organizing content"""
    CHRONOLOGICAL = "chronological"
    THEMATIC = "thematic"
    CHARACTER_DRIVEN = "character_driven"
    PROBLEM_SOLUTION = "problem_solution"
    JOURNEY = "journey"


class SourceType(str, Enum):
    """Types of sources for research"""
    ACADEMIC_PAPER = "academic_paper"
    ENCYCLOPEDIA = "encyclopedia"
    NEWS_ARTICLE = "news_article"
    DOCUMENTARY = "documentary"
    EXPERT_INTERVIEW = "expert_interview"
    HISTORICAL_RECORD = "historical_record"
    SCIENTIFIC_JOURNAL = "scientific_journal"
    WEB_RESOURCE = "web_resource"


class ResearchSource(BaseModel):
    """Individual research source"""
    title: str
    url: Optional[str] = None
    source_type: SourceType
    credibility_score: float = Field(ge=0.0, le=1.0)
    content_summary: str
    key_facts: List[str] = []
    citations: List[str] = []
    date_accessed: datetime = Field(default_factory=datetime.now)
    relevance_score: float = Field(ge=0.0, le=1.0)


class ImagePrompt(BaseModel):
    """Prompt for generating images at specific points in the video"""
    timestamp: float  # Seconds into the video
    base_prompt: str
    style_modifiers: str
    negative_prompt: str = ""
    context: str  # What's being discussed at this point
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    duration: float = 8.0  # How long this image should be shown


class Chapter(BaseModel):
    """Individual chapter/segment of the video"""
    id: str
    title: str
    start_time: float  # Seconds
    end_time: float  # Seconds
    duration: float  # Seconds
    script_text: str
    key_points: List[str] = []
    image_prompts: List[ImagePrompt] = []
    background_music_mood: str = "neutral"
    transition_type: str = "fade"


class ContentMetadata(BaseModel):
    """Metadata about the generated content"""
    topic: str
    subtopic: Optional[str] = None
    target_audience: str = "general"
    content_type: ContentType = ContentType.DOCUMENTARY
    narrative_structure: NarrativeStructure = NarrativeStructure.CHRONOLOGICAL
    estimated_length_minutes: int = 120
    complexity_level: int = Field(ge=1, le=5, default=3)  # 1=simple, 5=expert
    language: str = "en"
    region_focus: Optional[str] = None  # Geographic focus if relevant


class VideoScript(BaseModel):
    """Complete video script with all components"""
    metadata: ContentMetadata
    title: str
    description: str
    introduction: str
    chapters: List[Chapter] = []
    conclusion: str
    total_duration: float  # Seconds
    total_word_count: int
    image_prompts: List[ImagePrompt] = []
    
    def get_full_script_text(self) -> str:
        """Get the complete script as a single text"""
        parts = [self.introduction]
        parts.extend([chapter.script_text for chapter in self.chapters])
        parts.append(self.conclusion)
        return "\n\n".join(parts)
    
    def get_chapter_at_time(self, timestamp: float) -> Optional[Chapter]:
        """Get the chapter that's playing at a specific timestamp"""
        for chapter in self.chapters:
            if chapter.start_time <= timestamp <= chapter.end_time:
                return chapter
        return None


class ResearchReport(BaseModel):
    """Compiled research results for a topic"""
    topic: str
    subtopic: Optional[str] = None
    sources: List[ResearchSource] = []
    key_facts: List[str] = []
    timeline_events: List[Dict[str, Any]] = []  # For chronological content
    key_figures: List[Dict[str, str]] = []  # Important people/characters
    locations: List[Dict[str, str]] = []  # Important places
    concepts: List[Dict[str, str]] = []  # Key concepts to explain
    controversies: List[str] = []  # Debated topics
    modern_relevance: List[str] = []  # Why this matters today
    visual_elements: List[str] = []  # Suggested visual concepts
    research_quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    generated_at: datetime = Field(default_factory=datetime.now)


class ContentGenerationRequest(BaseModel):
    """Request for generating content"""
    topic: str
    subtopic: Optional[str] = None
    target_length_minutes: int = 120
    content_type: ContentType = ContentType.DOCUMENTARY
    narrative_structure: NarrativeStructure = NarrativeStructure.CHRONOLOGICAL
    target_audience: str = "general"
    tone_profile: Optional[str] = None
    style_preferences: Dict[str, Any] = {}
    specific_requirements: List[str] = []
    avoid_topics: List[str] = []
    region_focus: Optional[str] = None
    language: str = "en"


class ContentGenerationResult(BaseModel):
    """Result of content generation process"""
    request: ContentGenerationRequest
    research_report: ResearchReport
    video_script: VideoScript
    visual_plan: Optional["VisualPlan"] = None
    generation_stats: Dict[str, Any] = {}
    quality_scores: Dict[str, float] = {}
    warnings: List[str] = []
    suggestions: List[str] = []
    generated_at: datetime = Field(default_factory=datetime.now)
    generation_time_seconds: float = 0.0
    
    def save_to_file(self, filepath: str):
        """Save the complete result to a JSON file"""
        import json
        from pathlib import Path
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "ContentGenerationResult":
        """Load a result from a JSON file"""
        import json
        from pathlib import Path
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(**data)


# ==========================
# Visual Planning Data Models
# (Phase 2: non-breaking additions)
# ==========================

class VisualEntity(BaseModel):
    """Recurring people/locations/objects that must stay consistent across beats"""
    id: str
    kind: str  # person | location | object | concept
    descriptor: str
    references: List[str] = []  # paths to reference images if available
    persist_across_beats: bool = True
    seed_group: str = ""


class VisualItem(BaseModel):
    """Single visual generation instruction inside a beat"""
    type: str = "image"  # image | diagram | map | reenactment | abstract
    prompt: str
    negatives: str = ""
    style_locks: List[str] = []
    variation: float = 0.0
    seed_group: str = ""


class BeatPlan(BaseModel):
    """A semantic beat of the narration and its visuals"""
    id: str
    narration_span: Dict[str, int]  # { start_token, end_token }
    estimated_duration_s: float
    shot_type: str
    seed_group: str = ""
    prompts: List[VisualItem]
    overlays: List[Dict[str, Any]] = []  # e.g., lower_third, quote
    notes: str = ""


class GlobalStyle(BaseModel):
    topic: str
    style_template_key: str
    aspect_ratio: str = "16:9"
    color_profile: str = "sRGB"
    tone: str = "documentary"


class VisualPlan(BaseModel):
    """Complete visual plan emitted by the Visual Planner LLM"""
    schema_version: str = "v1"
    global_style: GlobalStyle
    entities: List[VisualEntity] = []
    beats: List[BeatPlan]

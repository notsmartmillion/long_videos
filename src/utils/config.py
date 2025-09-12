"""Configuration management for the video AI system"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    cuda_enabled: bool = True
    device: str = "cuda"
    max_memory_gb: int = 60
    batch_size: int = 4
    num_workers: int = 8


class ContentConfig(BaseModel):
    video_length_minutes: int = 120
    chapters: int = 8
    script_style: str = "documentary"
    target_audience: str = "general"
    narrative_structure: str = "chronological"


class TopicConfig(BaseModel):
    sources: List[str]
    visual_style: str
    keywords: List[str]


class LLMConfig(BaseModel):
    use_local_llm: bool = True
    local_llm_url: str = "http://localhost:11434/v1"
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: int = 4000


class ResearchConfig(BaseModel):
    enable_web_search: bool = True
    enable_academic_sources: bool = True
    enable_multimedia_sources: bool = True
    fact_check: bool = True
    citation_required: bool = True
    source_diversity: bool = True
    update_frequency: str = "daily"


class TTSConfig(BaseModel):
    engine: str = "coqui_xtts"
    voice_model: str = "v2"
    language: str = "en"
    speed: float = 1.05
    pitch: float = 0.0
    volume: float = 0.8
    output_quality: str = "high"


class StyleTemplate(BaseModel):
    base_style: str
    colors: str
    mood: str


class ImageGenerationConfig(BaseModel):
    engine: str = "flux"
    model_path: str = "./models/flux"
    resolution: str = "1920x1080"
    images_per_segment: int = 8
    batch_size: int = 4
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    style_templates: Dict[str, StyleTemplate] = {}
    
    def get_style_for_topic(self, topic: str) -> StyleTemplate:
        """Get style template for a specific topic"""
        return self.style_templates.get(topic, self.style_templates.get('default', 
            StyleTemplate(
                base_style="cinematic, high quality, professional",
                colors="balanced, natural",
                mood="engaging, clear, informative"
            )
        ))


class VisualPlannerConfig(BaseModel):
    model_name: str = "llama3.2"
    temperature: float = 0.2
    max_tokens: int = 6000
    output_schema_version: str = "v1"
    beat_target_seconds: List[int] = [6, 15]
    allow_abstract_shots: bool = True
    enabled: bool = True
    shot_types: List[str] = [
        "establishing", "medium_detail", "insert", "map", "diagram", "archival", "reenactment", "abstract"
    ]
    constraints: List[str] = []
    fallback_shots: List[str] = ["diagram", "map", "abstract"]


class ContinuityConfig(BaseModel):
    seed_strategy: str = "deterministic"
    seed_namespace: str = "doc_longform"
    entity_consistency: str = "strict"
    reuse_reference_images: bool = True
    variation_amount: float = 0.15
    caption_similarity_threshold: float = 0.62


class AlignmentConfig(BaseModel):
    forced_alignment: str = "aeneas"  # gentle | MFA | aeneas
    align_granularity: str = "sentence"  # word | phrase | sentence
    max_beat_length_seconds: int = 25
    split_on_punctuation: bool = True
    source: str = "heuristic"
    captioner: str = "stub"  # stub | blip | llava
    similarity_threshold: float = 0.62


class SimilarityConfig(BaseModel):
    mode: str = "stub"  # embeddings | stub


class VideoConfig(BaseModel):
    resolution: str = "1920x1080"
    fps: int = 30
    codec: str = "h264_nvenc"
    bitrate: str = "8M"
    ken_burns_effect: bool = True
    ken_burns_duration: int = 8
    transition_duration: int = 1
    background_music: bool = True
    music_volume: float = 0.15


class PathsConfig(BaseModel):
    """Storage paths configuration"""
    models: str = "./models"
    data: str = "./data"
    output: str = "./output"
    temp: str = "./temp"
    logs: str = "./logs"
    assets: str = "./assets"
    cache: str = "./temp/cache"


class Config(BaseModel):
    system: SystemConfig
    content: ContentConfig
    topics: Dict[str, TopicConfig] = {}
    llm: LLMConfig
    research: ResearchConfig
    tts: TTSConfig
    image_generation: ImageGenerationConfig
    video: VideoConfig
    paths: PathsConfig = PathsConfig()
    # New optional planning sections
    visual_planner: Optional[VisualPlannerConfig] = VisualPlannerConfig()
    continuity: Optional[ContinuityConfig] = ContinuityConfig()
    alignment: Optional[AlignmentConfig] = AlignmentConfig()
    similarity: Optional[SimilarityConfig] = SimilarityConfig()
    topic_adapters: Dict[str, Any] = {}
    
    # QA configuration (optional)
    class QAConfig(BaseModel):
        enabled: bool = True
        caption_similarity_threshold: float = 0.62
        max_retries: int = 1
        diversity_window: int = 8
        diversity_max_ratio: float = 0.4

    qa: Optional[QAConfig] = QAConfig()
    
    # Additional config sections that might be referenced
    youtube: Dict[str, Any] = {}
    automation: Dict[str, Any] = {}
    logging: Dict[str, Any] = {}
    performance: Dict[str, Any] = {}
    debug: Dict[str, Any] = {}
    
    def get_topic_config(self, topic: str) -> Optional[TopicConfig]:
        """Get configuration for a specific topic"""
        return self.topics.get(topic)
    
    def get_available_topics(self) -> List[str]:
        """Get list of configured topics"""
        return list(self.topics.keys())
    
    def is_topic_supported(self, topic: str) -> bool:
        """Check if a topic is supported"""
        return topic in self.topics
    
    @classmethod
    def load(cls, config_path: str) -> "Config":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save(self, config_path: str):
        """Save configuration to YAML file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)

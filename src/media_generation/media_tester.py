"""Test script for media generation components"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.media_generation.media_pipeline import MediaPipeline
from src.content_generation.content_models import VideoScript, Chapter, ImagePrompt, ContentMetadata


async def test_media_pipeline():
    """Test the media generation pipeline"""
    
    # Load config
    config = Config.load("configs/config.yaml")
    logger = setup_logging(config)
    
    logger.info("Testing Media Generation Pipeline")
    
    try:
        # Create test script
        test_script = create_test_script()
        
        # Initialize media pipeline
        media_pipeline = MediaPipeline(config)
        
        # Generate media
        logger.info("Generating test media...")
        result = await media_pipeline.generate_media(test_script, "mythology")
        
        # Display results
        logger.info("Media Generation Results:")
        logger.info(f"Audio segments: {len(result.audio_segments)}")
        logger.info(f"Generated images: {len(result.generated_images)}")
        logger.info(f"Total duration: {result.total_audio_duration:.1f}s")
        logger.info(f"Generation time: {result.total_generation_time:.1f}s")
        
        # Validate quality
        validation = await media_pipeline.validate_media_quality(result)
        logger.info(f"Quality validation passed: {validation['passed']}")
        
        if validation['issues']:
            logger.warning(f"Issues: {validation['issues']}")
        if validation['warnings']:
            logger.warning(f"Warnings: {validation['warnings']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Media pipeline test failed: {e}")
        raise


def create_test_script() -> VideoScript:
    """Create a test video script"""
    
    metadata = ContentMetadata(
        topic="mythology",
        subtopic="Zeus",
        estimated_length_minutes=5  # Short test
    )
    
    # Test chapters
    chapters = [
        Chapter(
            id="chapter_1",
            title="The Birth of Zeus",
            start_time=0.0,
            end_time=120.0,  # 2 minutes
            duration=120.0,
            script_text="In the ancient times of Greek mythology, Zeus was born to the Titan Cronus and Rhea. This is the story of how the king of the gods came to power, overthrowing his father and establishing the reign of the Olympians on Mount Olympus.",
            key_points=["Zeus's birth", "Cronus", "Olympians"],
            image_prompts=[
                ImagePrompt(
                    timestamp=10.0,
                    base_prompt="Zeus as a baby being hidden from Cronus",
                    style_modifiers="classical art, dramatic lighting",
                    context="Zeus's birth story"
                ),
                ImagePrompt(
                    timestamp=60.0,
                    base_prompt="Mount Olympus in classical Greek art style",
                    style_modifiers="majestic, divine, golden lighting",
                    context="The realm of the gods"
                )
            ]
        ),
        Chapter(
            id="chapter_2", 
            title="Zeus Becomes King",
            start_time=120.0,
            end_time=240.0,
            duration=120.0,
            script_text="After growing to maturity, Zeus challenged his father Cronus in an epic battle known as the Titanomachy. With the help of his siblings and the Cyclopes, Zeus emerged victorious and became the ruler of the heavens.",
            key_points=["Titanomachy", "Victory", "King of gods"],
            image_prompts=[
                ImagePrompt(
                    timestamp=150.0,
                    base_prompt="Epic battle between Zeus and Cronus with lightning",
                    style_modifiers="dramatic, powerful, stormy",
                    context="The Titanomachy battle"
                ),
                ImagePrompt(
                    timestamp=210.0,
                    base_prompt="Zeus sitting on his throne with thunderbolt",
                    style_modifiers="regal, authoritative, divine power",
                    context="Zeus as king of gods"
                )
            ]
        )
    ]
    
    # Additional image prompts
    image_prompts = [
        ImagePrompt(
            timestamp=5.0,
            base_prompt="Ancient Greek temple with columns",
            style_modifiers="classical architecture, golden hour",
            context="Introduction setting"
        )
    ]
    
    # Combine all image prompts
    all_image_prompts = image_prompts.copy()
    for chapter in chapters:
        all_image_prompts.extend(chapter.image_prompts)
    
    script = VideoScript(
        metadata=metadata,
        title="Zeus: The Rise of the King of Gods",
        description="A brief documentary about Zeus's journey to becoming king of the Olympian gods",
        introduction="Welcome to the epic tale of Zeus, the mighty king of the Greek gods. Today we explore how this powerful deity rose from humble beginnings to rule Mount Olympus.",
        chapters=chapters,
        conclusion="And so Zeus established his reign over the heavens, becoming the most powerful of all the Greek gods. His stories continue to inspire us thousands of years later.",
        total_duration=300.0,  # 5 minutes
        total_word_count=150,
        image_prompts=all_image_prompts
    )
    
    return script


async def test_individual_components():
    """Test TTS and image generation separately"""
    
    config = Config.load("configs/config.yaml")
    logger = setup_logging(config)
    
    # Test TTS Engine
    logger.info("Testing TTS Engine...")
    from src.media_generation.tts_engine import TTSEngine
    from src.media_generation.media_models import AudioGenerationRequest, AudioQuality
    
    tts_engine = TTSEngine(config)
    await tts_engine.initialize()
    
    test_request = AudioGenerationRequest(
        script_text="This is a test of the text-to-speech system. Zeus was the king of the gods in Greek mythology.",
        voice_model="documentary_narrator",
        quality=AudioQuality.HIGH
    )
    
    audio_segments = await tts_engine.generate_audio(test_request, "mythology")
    logger.info(f"Generated {len(audio_segments)} audio segments")
    
    # Test Image Generator
    logger.info("Testing Image Generator...")
    from src.media_generation.image_generator import ImageGenerator
    
    image_generator = ImageGenerator(config)
    await image_generator.initialize()
    
    test_prompts = [
        "Zeus sitting on his throne with lightning bolt",
        "Ancient Greek temple with marble columns"
    ]
    
    generated_images = await image_generator.generate_images(
        test_prompts, "mythology", [0.0, 10.0]
    )
    logger.info(f"Generated {len(generated_images)} images")
    
    return audio_segments, generated_images


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test media generation components")
    parser.add_argument("--component", choices=["all", "tts", "images", "pipeline"], 
                       default="pipeline", help="Which component to test")
    
    args = parser.parse_args()
    
    if args.component == "pipeline":
        asyncio.run(test_media_pipeline())
    elif args.component == "all":
        asyncio.run(test_individual_components())
    else:
        print(f"Testing {args.component} component...")
        # Individual component tests would go here


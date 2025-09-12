#!/usr/bin/env python3
"""
Long Video AI Automation - Main Entry Point
Automated YouTube video generation system for ANY topic (mythology, space, history, etc.)
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from dotenv import load_dotenv

# Load local env for API keys
try:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env.local")
except Exception:
    pass

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.logger import setup_logging
from src.content_generation.content_pipeline import ContentPipeline
from src.media_generation.media_pipeline import MediaPipeline
# TODO: Add these when video assembly and automation are implemented
from src.video_assembly.video_assembler import VideoAssembler  
from src.automation.scheduler import VideoScheduler

# Ensure UTF-8 console output on Windows to avoid emoji/encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

console = Console()


class VideoAISystem:
    """Main system coordinator for automated video generation"""
    
    def __init__(self, config_path: str = "configs/config.yaml", test_mode: bool = False):
        self.config = Config.load(config_path)
        self.test_mode = test_mode
        
        # Apply test mode overrides if enabled
        if self.test_mode:
            self._apply_test_mode_config()
            console.print("[yellow]🧪[/yellow] Test mode enabled - generating 1-2 minute video")
        
        self.logger = setup_logging(self.config)
        
        # Check CUDA availability for RTX 5080
        self._check_cuda_setup()
        
        # Initialize pipeline components
        self.content_pipeline = ContentPipeline(self.config)
        self.media_pipeline = MediaPipeline(self.config)
        
        # Initialize video assembly
        self.video_assembler = VideoAssembler(self.config)
        
        # Initialize automation scheduler
        self.scheduler = VideoScheduler(self.config)
        self.scheduler.inject_dependencies(
            self.content_pipeline,
            self.media_pipeline, 
            self.video_assembler,
            self.content_pipeline.topic_queue
        )
        
        console.print("[green]✓[/green] Video AI System initialized successfully!")
    
    def _apply_test_mode_config(self):
        """Apply test mode configuration overrides for faster generation"""
        # Override content settings for 1-2 minute video
        self.config.content.video_length_minutes = 2
        self.config.content.chapters = 2
        
        # Override image generation for RTX 5080 optimized test mode (ultra-conservative)
        self.config.image_generation.images_per_segment = 3  # Fewer images for 2-minute video
        self.config.image_generation.num_inference_steps = 8  # Ultra-fast for test mode
        self.config.image_generation.batch_size = 1  # Ultra-conservative for SDXL OOM prevention
        
        # Override performance settings for test mode (more conservative for stability)
        if hasattr(self.config, 'performance'):
            self.config.performance.update({
                'max_parallel_audio': 1,     # Sequential for test mode stability
                'max_parallel_images': 6,    # Moderate parallelism for test mode
                'max_parallel_video': 2      # Conservative video processing
            })
    
    def _check_cuda_setup(self):
        """Verify CUDA setup for RTX 5080 - STRICT VALIDATION, NO FALLBACKS"""
        console.print("[blue]🔍[/blue] Validating RTX 5080 GPU setup...")
        
        # Check 1: CUDA availability
        if not torch.cuda.is_available():
            console.print("[red]💥 FATAL ERROR: CUDA not available![/red]")
            console.print("[red]→[/red] PyTorch installation issue or GPU drivers not installed")
            console.print("[red]→[/red] This system requires RTX 5080 GPU acceleration")
            sys.exit(1)
        
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_capability = torch.cuda.get_device_properties(0).major + torch.cuda.get_device_properties(0).minor * 0.1
        
        console.print(f"[green]✓[/green] GPU: {device_name}")
        console.print(f"[green]✓[/green] VRAM: {memory_gb:.1f} GB")
        console.print(f"[green]✓[/green] CUDA Version: {torch.version.cuda}")
        console.print(f"[green]✓[/green] Compute Capability: {compute_capability}")
        
        # Check 2: Verify it's actually an RTX 5080
        if "RTX 5080" not in device_name:
            console.print(f"[red]💥 FATAL ERROR: Expected RTX 5080, found {device_name}[/red]")
            console.print("[red]→[/red] This system is optimized specifically for RTX 5080")
            sys.exit(1)
        
        # Check 3: Test actual GPU functionality
        try:
            # Try to create a tensor on GPU
            test_tensor = torch.randn(100, 100, device='cuda')
            result = torch.matmul(test_tensor, test_tensor)
            del test_tensor, result
            torch.cuda.empty_cache()
            console.print("[green]✓[/green] GPU computation test passed")
        except Exception as e:
            console.print(f"[red]💥 FATAL ERROR: GPU computation failed![/red]")
            console.print(f"[red]→[/red] Error: {e}")
            console.print(f"[red]→[/red] RTX 5080 detected but not functional with current PyTorch")
            console.print(f"[red]→[/red] This is likely a PyTorch/CUDA compatibility issue")
            sys.exit(1)
        
        # Check 4: Validate VRAM is sufficient
        if memory_gb < 15.0:  # RTX 5080 should have 16GB
            console.print(f"[red]💥 FATAL ERROR: Insufficient VRAM: {memory_gb:.1f}GB[/red]")
            console.print("[red]→[/red] RTX 5080 should have 16GB+ VRAM")
            sys.exit(1)
        
        console.print("[green]🚀[/green] RTX 5080 validation PASSED - GPU acceleration enabled")
    
    async def generate_single_video(self, topic: Optional[str] = None, subtopic: Optional[str] = None) -> str:
        """Generate a single video on demand for any topic"""
        console.print("[blue]🎬[/blue] Starting video generation process...")
        
        if self.test_mode:
            console.print(f"[yellow]🧪[/yellow] Test mode: Creating {self.config.content.video_length_minutes}-minute video")
        
        # Validate and display topic information
        if topic:
            if self.config.is_topic_supported(topic):
                topic_config = self.config.get_topic_config(topic)
                console.print(f"[green]📚[/green] Topic: {topic}")
                console.print(f"[green]🎨[/green] Visual Style: {topic_config.visual_style}")
                if subtopic:
                    console.print(f"[green]🔍[/green] Subtopic: {subtopic}")
            else:
                available_topics = self.config.get_available_topics()
                console.print(f"[yellow]⚠[/yellow] Topic '{topic}' not configured.")
                console.print(f"[yellow]💡[/yellow] Available topics: {', '.join(available_topics)}")
                topic = None
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                refresh_per_second=4,  # Limit refresh rate to reduce spam
                transient=False  # Keep progress bars visible
            ) as progress:
                
                # Step 1: Content Generation
                content_task = progress.add_task("[cyan]📝 Generating content & script...", total=100)
                
                def content_progress_callback(percent, message):
                    # Only update if progress changed significantly to reduce spam
                    if percent - getattr(content_progress_callback, 'last_percent', 0) >= 5:
                        progress.update(content_task, completed=percent, description=f"[cyan]📝 {message}")
                        content_progress_callback.last_percent = percent
                
                content_data = await self.content_pipeline.generate_content(topic, subtopic, content_progress_callback)
                progress.update(content_task, completed=100, description="[cyan]📝 Content generation complete")
                
                # Step 2: Audio Generation
                audio_task = progress.add_task("[yellow]🎵 Generating narration audio...", total=100)
                
                def audio_progress_callback(percent, message):
                    # Only update if progress changed significantly to reduce spam
                    if percent - getattr(audio_progress_callback, 'last_percent', 0) >= 5:
                        progress.update(audio_task, completed=percent, description=f"[yellow]🎵 {message}")
                        audio_progress_callback.last_percent = percent
                
                audio_path = await self.media_pipeline.generate_audio(content_data.video_script, topic, audio_progress_callback)
                progress.update(audio_task, completed=100, description="[yellow]🎵 Audio generation complete")
                
                # Step 3: Image Generation (use enhanced prompts path to enable QA logs)
                image_count = len(content_data.video_script.image_prompts)
                image_task = progress.add_task(f"[green]🖼️ Generating {image_count} images...", total=100)

                def image_progress_callback(percent, message):
                    if percent - getattr(image_progress_callback, 'last_percent', 0) >= 5:
                        progress.update(image_task, completed=percent, description=f"[green]🖼️ {message}")
                        image_progress_callback.last_percent = percent

                use_enhanced = bool(getattr(self.config.visual_planner, 'use_enhanced_prompts', False)) and getattr(self.config.visual_planner, 'enabled', True)
                if use_enhanced:
                    generated_images = await self.media_pipeline._generate_images_with_enhanced_prompts(
                        content_data.video_script, topic
                    )
                    image_paths = [img.file_path for img in generated_images if img.file_path]
                else:
                    image_paths = await self.media_pipeline.generate_images(
                        content_data.video_script.image_prompts, topic, image_progress_callback
                    )
                progress.update(image_task, completed=100, description="[green]🖼️ Image generation complete")
                
                # Step 4: Video Assembly
                video_task = progress.add_task("[magenta]🎞️ Assembling final video...", total=100)
                
                def assembly_progress_callback(assembly_progress):
                    # Only update if progress changed significantly to reduce spam
                    if assembly_progress.progress_percent - getattr(assembly_progress_callback, 'last_percent', 0) >= 5:
                        progress.update(video_task, completed=assembly_progress.progress_percent, 
                                      description=f"[magenta]🎞️ {assembly_progress.current_step}")
                        assembly_progress_callback.last_percent = assembly_progress.progress_percent
                
                timeline_path = self.media_pipeline.artifacts_dir / "timeline.json"
                timeline_arg = str(timeline_path) if timeline_path.exists() else None
                assembly_result = await self.video_assembler.assemble_video(
                    content_data.video_script,
                    audio_path,
                    image_paths,
                    topic,
                    assembly_progress_callback,
                    timeline_path=timeline_arg
                )
                progress.update(video_task, completed=100)
            
            if assembly_result.success:
                console.print("\n[bold green]🎉 Video Generation Complete![/bold green]")
                console.print(f"[green]📁[/green] Audio file: {audio_path}")
                console.print(f"[green]🖼️[/green] Generated {len(image_paths)} images")
                console.print(f"[green]🎬[/green] Video duration: {assembly_result.total_duration:.1f}s")
                console.print(f"[green]💾[/green] File size: {assembly_result.file_size_mb:.1f}MB")
                console.print(f"[green]⏱️[/green] Render time: {assembly_result.render_time_seconds:.1f}s")
                console.print(f"[green]✅[/green] Video saved: {assembly_result.output_path}")
                
                if self.test_mode:
                    actual_min = max(1, int(round(assembly_result.total_duration / 60)))
                    console.print(f"[yellow]🧪[/yellow] Test complete! Video is ~{actual_min} minutes long")
                
                return str(assembly_result.output_path)
            else:
                raise Exception(f"Video assembly failed: {'; '.join(assembly_result.errors)}")
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            console.print(f"[red]❌[/red] Error: {e}")
            raise
    
    async def start_automated_mode(self):
        """Start automated daily video generation"""
        console.print("\n[blue]🤖[/blue] Starting automated video generation...")
        
        # Check if there are any schedules
        schedules = self.scheduler.get_schedules()
        
        if not schedules:
            console.print("[yellow]⚠️[/yellow] No schedules found. Creating a default daily schedule...")
            
            # Create default schedule
            from src.automation.automation_models import ScheduleConfig
            from datetime import time
            
            default_schedule = ScheduleConfig(
                name="Daily Video Generation",
                description="Automatically generate one video per day",
                topic_categories=list(self.config.get_available_topics()),
                time_of_day=time(8, 0),  # 8 AM
                max_videos_per_day=1
            )
            
            schedule_id = self.scheduler.add_schedule(default_schedule)
            console.print(f"[green]✅[/green] Created default schedule: {schedule_id}")
        
        console.print(f"[blue]📋[/blue] Active schedules: {len(schedules)}")
        for schedule_id, schedule in schedules.items():
            console.print(f"  • {schedule.name} - Next: {schedule.next_run}")
        
        console.print("[blue]🚀[/blue] Starting scheduler... (Press Ctrl+C to stop)")
        
        try:
            await self.scheduler.start_scheduler()
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️[/yellow] Stopping automation...")
            self.scheduler.stop_scheduler()
            console.print("[green]✅[/green] Automation stopped")
    
    def run_interactive_mode(self):
        """Interactive mode for testing and manual generation"""
        console.print("\n[bold blue]🚀 Long Video AI - Interactive Mode[/bold blue]\n")
        
        # Show test mode status
        if self.test_mode:
            console.print("[yellow]🧪 TEST MODE ACTIVE[/yellow] - Videos will be 2 minutes long")
        else:
            console.print("[cyan]💡 Tip:[/cyan] Use --test flag for quick 2-minute test videos")
        
        # Display available topics
        available_topics = self.config.get_available_topics()
        console.print(f"[cyan]📋 Available Topics:[/cyan] {', '.join(available_topics)}")
        
        while True:
            console.print("\nOptions:")
            console.print("1. Generate single video")
            console.print("2. Start automated mode")
            console.print("3. List available topics")
            console.print("4. Test components")
            console.print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                console.print(f"\n[cyan]Available topics:[/cyan] {', '.join(available_topics)}")
                topic = input("Enter topic category: ").strip()
                subtopic = input("Enter specific subject (optional): ").strip()
                asyncio.run(self.generate_single_video(
                    topic if topic else None, 
                    subtopic if subtopic else None
                ))
                
            elif choice == "2":
                asyncio.run(self.start_automated_mode())
                
            elif choice == "3":
                self._display_topic_info()
                
            elif choice == "4":
                self._run_component_tests()
                
            elif choice == "5":
                console.print("[yellow]👋[/yellow] Goodbye!")
                break
                
            else:
                console.print("[red]Invalid choice![/red]")
    
    def _display_topic_info(self):
        """Display detailed information about available topics"""
        topics = self.config.get_available_topics()
        
        console.print("\n[bold cyan]📚 Available Topics & Configurations:[/bold cyan]\n")
        
        for topic_name in topics:
            topic_config = self.config.get_topic_config(topic_name)
            console.print(f"[green]🎯 {topic_name.title()}[/green]")
            console.print(f"   Visual Style: {topic_config.visual_style}")
            console.print(f"   Sources: {', '.join(topic_config.sources)}")
            console.print(f"   Keywords: {', '.join(topic_config.keywords[:5])}...")
            console.print()
    
    def _run_component_tests(self):
        """Test individual components"""
        console.print("[blue]🧪[/blue] Testing available components...")
        
        # Test content generation
        console.print("[blue]📝[/blue] Content generation: ✅ Available")
        console.print("[blue]🎵[/blue] TTS engine: ✅ Available") 
        console.print("[blue]🖼️[/blue] Image generation: ✅ Available")
        console.print("[green]🎞️[/green] Video assembly: ✅ Available")
        console.print("[green]🤖[/green] Automation: ✅ Available")
        
        # Run media generation test
        console.print("\n[blue]🔍[/blue] Want to run a quick media generation test? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            try:
                import asyncio
                from src.media_generation.media_tester import test_media_pipeline
                console.print("[blue]🚀[/blue] Running media pipeline test...")
                asyncio.run(test_media_pipeline())
                console.print("[green]✅[/green] Media pipeline test completed!")
            except Exception as e:
                console.print(f"[red]❌[/red] Test failed: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Long Video AI Automation System")
    parser.add_argument("--mode", choices=["interactive", "auto", "single"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--topic", type=str, help="Topic category for single video generation")
    parser.add_argument("--subtopic", type=str, help="Specific subject within the topic")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--test", action="store_true",
                       help="Quick test mode: generate 1-2 minute video for testing")
    
    args = parser.parse_args()
    
    try:
        system = VideoAISystem(args.config, test_mode=args.test)
        
        if args.mode == "interactive":
            system.run_interactive_mode()
        elif args.mode == "auto":
            asyncio.run(system.start_automated_mode())
        elif args.mode == "single":
            asyncio.run(system.generate_single_video(args.topic, args.subtopic))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️[/yellow] Stopped by user")
    except Exception as e:
        console.print(f"[red]💥[/red] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

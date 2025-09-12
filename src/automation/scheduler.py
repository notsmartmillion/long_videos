"""
Video Scheduler

Automated video generation scheduler with queue management.
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import yaml
from threading import Lock
import psutil
import shutil

from .automation_models import (
    ScheduleConfig, VideoJob, AutomationResult, AutomationStats,
    HealthCheck, QueueMetrics, ScheduleStatus, ScheduleFrequency
)
from ..content_generation.topic_queue import TopicQueue

class VideoScheduler:
    """
    Automated video generation scheduler.
    
    Features:
    - Configurable scheduling (daily, weekly, monthly)
    - Queue management with retries
    - Health monitoring
    - Resource management
    - Progress tracking
    - Error recovery
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.schedules_dir = Path(getattr(config.paths, 'data', './data')) / 'schedules'
        self.jobs_dir = Path(getattr(config.paths, 'data', './data')) / 'jobs'
        self.logs_dir = Path(getattr(config.paths, 'logs', './logs'))
        
        # Create directories
        for dir_path in [self.schedules_dir, self.jobs_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # State management
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.active_jobs: Dict[str, VideoJob] = {}
        self.job_history: List[VideoJob] = []
        self.is_running = False
        self.state_lock = Lock()
        
        # Dependencies (injected)
        self.content_pipeline = None
        self.media_pipeline = None
        self.video_assembler = None
        self.topic_queue = None
        
        # Performance tracking
        self.stats = AutomationStats()
        self.start_time = datetime.now()
        
        # Load existing schedules and jobs
        self._load_schedules()
        self._load_jobs()
        
        self.logger.info("Video scheduler initialized")
    
    def inject_dependencies(self, content_pipeline, media_pipeline, video_assembler, topic_queue):
        """Inject pipeline dependencies"""
        self.content_pipeline = content_pipeline
        self.media_pipeline = media_pipeline
        self.video_assembler = video_assembler
        self.topic_queue = topic_queue
        self.logger.info("Dependencies injected into scheduler")
    
    def add_schedule(self, schedule: ScheduleConfig) -> str:
        """Add a new schedule"""
        schedule_id = str(uuid.uuid4())
        
        # Calculate next run time
        schedule.next_run = self._calculate_next_run(schedule)
        
        with self.state_lock:
            self.schedules[schedule_id] = schedule
        
        # Save to disk
        self._save_schedule(schedule_id, schedule)
        
        self.logger.info(f"Added schedule '{schedule.name}' (ID: {schedule_id})")
        return schedule_id
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule"""
        with self.state_lock:
            if schedule_id in self.schedules:
                del self.schedules[schedule_id]
                
                # Remove file
                schedule_file = self.schedules_dir / f"{schedule_id}.yaml"
                if schedule_file.exists():
                    schedule_file.unlink()
                
                self.logger.info(f"Removed schedule {schedule_id}")
                return True
        
        return False
    
    def get_schedules(self) -> Dict[str, ScheduleConfig]:
        """Get all schedules"""
        with self.state_lock:
            return self.schedules.copy()
    
    def get_active_jobs(self) -> Dict[str, VideoJob]:
        """Get currently active jobs"""
        with self.state_lock:
            return self.active_jobs.copy()
    
    def get_job_history(self, limit: int = 100) -> List[VideoJob]:
        """Get recent job history"""
        with self.state_lock:
            return self.job_history[-limit:].copy()
    
    async def start_scheduler(self) -> None:
        """Start the automated scheduler"""
        if self.is_running:
            self.logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.logger.info("Starting video scheduler...")
        
        try:
            while self.is_running:
                # Check for scheduled jobs
                await self._check_schedules()
                
                # Process pending jobs
                await self._process_job_queue()
                
                # Cleanup old files
                await self._cleanup_old_files()
                
                # Update statistics
                self._update_stats()
                
                # Health check
                await self._perform_health_check()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
        finally:
            self.is_running = False
            self.logger.info("Scheduler stopped")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler"""
        self.is_running = False
        self.logger.info("Stopping scheduler...")
    
    async def generate_video_manual(self, 
                                  topic_category: str,
                                  subtopic: Optional[str] = None,
                                  quality_preset: str = "balanced") -> AutomationResult:
        """Manually trigger video generation"""
        
        if not self._check_dependencies():
            return AutomationResult(
                success=False,
                message="Dependencies not available"
            )
        
        # Create job
        job_id = str(uuid.uuid4())
        job = VideoJob(
            id=job_id,
            schedule_id="manual",
            topic_category=topic_category,
            topic_title=subtopic or f"Random {topic_category} topic",
            subtopic=subtopic,
            status=ScheduleStatus.PENDING
        )
        
        with self.state_lock:
            self.active_jobs[job_id] = job
        
        # Execute job
        return await self._execute_job(job)
    
    async def _check_schedules(self) -> None:
        """Check for schedules that need to run"""
        now = datetime.now()
        
        with self.state_lock:
            schedules_to_run = []
            
            for schedule_id, schedule in self.schedules.items():
                if (schedule.enabled and 
                    schedule.next_run and 
                    schedule.next_run <= now):
                    schedules_to_run.append((schedule_id, schedule))
        
        # Execute scheduled jobs
        for schedule_id, schedule in schedules_to_run:
            await self._execute_schedule(schedule_id, schedule)
    
    async def _execute_schedule(self, schedule_id: str, schedule: ScheduleConfig) -> None:
        """Execute a scheduled job"""
        try:
            # Check if we can run more jobs
            with self.state_lock:
                running_count = sum(1 for job in self.active_jobs.values() 
                                  if job.status == ScheduleStatus.RUNNING)
            
            if running_count >= schedule.max_concurrent_videos:
                self.logger.info(f"Delaying schedule {schedule_id} - max concurrent limit reached")
                return
            
            # Get topics for this schedule
            topics = await self._get_topics_for_schedule(schedule)
            
            if not topics:
                self.logger.warning(f"No topics available for schedule {schedule_id}")
                # Update next run time anyway
                schedule.next_run = self._calculate_next_run(schedule)
                return
            
            # Create jobs for videos
            jobs_created = 0
            for topic_data in topics[:schedule.max_videos_per_day]:
                job_id = str(uuid.uuid4())
                job = VideoJob(
                    id=job_id,
                    schedule_id=schedule_id,
                    topic_category=topic_data.get('category', 'general'),
                    topic_title=topic_data.get('title', 'Unknown'),
                    subtopic=topic_data.get('subtopic'),
                    status=ScheduleStatus.PENDING
                )
                
                with self.state_lock:
                    self.active_jobs[job_id] = job
                
                jobs_created += 1
                self.logger.info(f"Created job {job_id} for schedule {schedule_id}")
            
            # Update schedule
            schedule.last_run = datetime.now()
            schedule.next_run = self._calculate_next_run(schedule)
            self._save_schedule(schedule_id, schedule)
            
            self.logger.info(f"Executed schedule {schedule_id}, created {jobs_created} jobs")
            
        except Exception as e:
            self.logger.error(f"Failed to execute schedule {schedule_id}: {e}")
    
    async def _process_job_queue(self) -> None:
        """Process pending jobs in the queue"""
        with self.state_lock:
            pending_jobs = [job for job in self.active_jobs.values() 
                          if job.status == ScheduleStatus.PENDING]
        
        # Process jobs concurrently (limited by config)
        max_concurrent = getattr(self.config.automation, 'max_concurrent_jobs', 2)
        running_count = sum(1 for job in self.active_jobs.values() 
                          if job.status == ScheduleStatus.RUNNING)
        
        available_slots = max_concurrent - running_count
        
        if available_slots > 0 and pending_jobs:
            # Sort by creation time (FIFO)
            pending_jobs.sort(key=lambda j: j.created_at)
            
            # Execute jobs
            tasks = []
            for job in pending_jobs[:available_slots]:
                task = asyncio.create_task(self._execute_job(job))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_job(self, job: VideoJob) -> AutomationResult:
        """Execute a single video generation job"""
        start_time = time.time()
        
        try:
            # Update job status
            job.status = ScheduleStatus.RUNNING
            job.started_at = datetime.now()
            job.current_step = "starting"
            self._save_job(job)
            
            self.logger.info(f"Starting job {job.id}: {job.topic_title}")
            
            # Step 1: Content Generation
            job.current_step = "generating_content"
            job.progress_percent = 10.0
            self._save_job(job)
            
            content_data = await self.content_pipeline.generate_content(
                job.topic_category, job.subtopic
            )
            
            # Step 2: Media Generation
            job.current_step = "generating_media"
            job.progress_percent = 40.0
            self._save_job(job)
            
            audio_path = await self.media_pipeline.generate_audio(
                content_data.video_script, job.topic_category
            )
            
            image_paths = await self.media_pipeline.generate_images(
                content_data.video_script.image_prompts, job.topic_category
            )
            
            # Step 3: Video Assembly
            job.current_step = "assembling_video"
            job.progress_percent = 70.0
            self._save_job(job)
            
            def progress_callback(progress):
                job.progress_percent = 70.0 + (progress.progress_percent * 0.25)
                job.current_step = f"assembling: {progress.current_step}"
                self._save_job(job)
            
            assembly_result = await self.video_assembler.assemble_video(
                content_data.video_script,
                audio_path,
                image_paths,
                job.topic_category,
                progress_callback
            )
            
            if not assembly_result.success:
                raise Exception(f"Video assembly failed: {'; '.join(assembly_result.errors)}")
            
            # Update job with results
            job.status = ScheduleStatus.COMPLETED
            job.completed_at = datetime.now()
            job.current_step = "completed"
            job.progress_percent = 100.0
            job.video_path = assembly_result.output_path
            job.thumbnail_path = assembly_result.thumbnail_path
            job.audio_path = audio_path
            job.render_time_seconds = assembly_result.render_time_seconds
            job.file_size_mb = assembly_result.file_size_mb
            job.video_duration_seconds = assembly_result.total_duration
            
            # Move to history
            with self.state_lock:
                if job.id in self.active_jobs:
                    del self.active_jobs[job.id]
                self.job_history.append(job)
            
            self._save_job(job)
            
            # Create result
            result = AutomationResult(
                success=True,
                message=f"Video generated successfully: {job.topic_title}",
                job_id=job.id,
                schedule_id=job.schedule_id,
                video_path=job.video_path,
                thumbnail_path=job.thumbnail_path,
                execution_time_seconds=time.time() - start_time
            )
            
            self.logger.info(f"Job {job.id} completed successfully in {result.execution_time_seconds:.1f}s")
            return result
            
        except Exception as e:
            # Handle failure
            job.status = ScheduleStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            job.retry_count += 1
            
            # Move to history or retry
            if job.retry_count >= job.max_retries:
                with self.state_lock:
                    if job.id in self.active_jobs:
                        del self.active_jobs[job.id]
                    self.job_history.append(job)
                
                self.logger.error(f"Job {job.id} failed permanently: {e}")
            else:
                # Reset for retry
                job.status = ScheduleStatus.PENDING
                job.current_step = "retry_queued"
                job.started_at = None
                job.progress_percent = 0.0
                
                self.logger.warning(f"Job {job.id} failed, will retry ({job.retry_count}/{job.max_retries}): {e}")
            
            self._save_job(job)
            
            return AutomationResult(
                success=False,
                message=f"Video generation failed: {e}",
                job_id=job.id,
                schedule_id=job.schedule_id,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _get_topics_for_schedule(self, schedule: ScheduleConfig) -> List[Dict[str, Any]]:
        """Get topics for a schedule"""
        topics = []
        
        # Use topic queue if available
        if self.topic_queue:
            for category in schedule.topic_categories:
                topic_data = self.topic_queue.get_next_topic(category)
                if topic_data:
                    topics.append({
                        'category': category,
                        'title': topic_data.get('title', 'Unknown'),
                        'subtopic': topic_data.get('subtopic')
                    })
        
        # Fallback: generate generic topics
        if not topics:
            for category in schedule.topic_categories:
                topics.append({
                    'category': category,
                    'title': f"Daily {category} topic",
                    'subtopic': None
                })
        
        return topics
    
    def _calculate_next_run(self, schedule: ScheduleConfig) -> datetime:
        """Calculate next run time for a schedule"""
        now = datetime.now()
        target_time = schedule.time_of_day
        
        if schedule.frequency == ScheduleFrequency.DAILY:
            next_run = now.replace(
                hour=target_time.hour,
                minute=target_time.minute,
                second=0,
                microsecond=0
            )
            
            # If time has passed today, schedule for tomorrow
            if next_run <= now:
                next_run += timedelta(days=1)
                
        elif schedule.frequency == ScheduleFrequency.WEEKLY:
            # Schedule for same day next week
            next_run = now.replace(
                hour=target_time.hour,
                minute=target_time.minute,
                second=0,
                microsecond=0
            ) + timedelta(weeks=1)
            
        elif schedule.frequency == ScheduleFrequency.MONTHLY:
            # Schedule for same day next month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1)
            else:
                next_month = now.replace(month=now.month + 1)
            
            next_run = next_month.replace(
                hour=target_time.hour,
                minute=target_time.minute,
                second=0,
                microsecond=0
            )
        else:
            # ONCE - don't schedule again
            return None
        
        return next_run
    
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        return all([
            self.content_pipeline is not None,
            self.media_pipeline is not None,
            self.video_assembler is not None
        ])
    
    async def _cleanup_old_files(self) -> None:
        """Clean up old generated files"""
        try:
            # Get cleanup threshold
            cleanup_days = getattr(self.config.automation, 'cleanup_after_days', 30)
            cutoff_date = datetime.now() - timedelta(days=cleanup_days)
            
            # Clean up old jobs
            old_jobs = [job for job in self.job_history 
                       if job.completed_at and job.completed_at < cutoff_date]
            
            for job in old_jobs:
                # Remove video files
                for path in [job.video_path, job.thumbnail_path, job.audio_path]:
                    if path and Path(path).exists():
                        try:
                            Path(path).unlink()
                        except Exception as e:
                            self.logger.warning(f"Failed to delete {path}: {e}")
                
                # Remove from history
                if job in self.job_history:
                    self.job_history.remove(job)
            
            if old_jobs:
                self.logger.info(f"Cleaned up {len(old_jobs)} old jobs")
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def _update_stats(self) -> None:
        """Update automation statistics"""
        with self.state_lock:
            # Count jobs
            self.stats.total_jobs = len(self.job_history) + len(self.active_jobs)
            self.stats.completed_jobs = sum(1 for job in self.job_history 
                                          if job.status == ScheduleStatus.COMPLETED)
            self.stats.failed_jobs = sum(1 for job in self.job_history 
                                       if job.status == ScheduleStatus.FAILED)
            self.stats.pending_jobs = sum(1 for job in self.active_jobs.values() 
                                        if job.status == ScheduleStatus.PENDING)
            
            # Calculate averages
            completed_jobs = [job for job in self.job_history 
                            if job.status == ScheduleStatus.COMPLETED and job.render_time_seconds]
            
            if completed_jobs:
                self.stats.average_render_time = sum(job.render_time_seconds for job in completed_jobs) / len(completed_jobs)
                self.stats.average_file_size_mb = sum(job.file_size_mb or 0 for job in completed_jobs) / len(completed_jobs)
            
            # Success rate
            if self.stats.total_jobs > 0:
                self.stats.success_rate_percent = (self.stats.completed_jobs / self.stats.total_jobs) * 100
            
            # Recent activity
            if completed_jobs:
                self.stats.last_job_completion = max(job.completed_at for job in completed_jobs)
            
            # System uptime
            self.stats.system_uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
    
    async def _perform_health_check(self) -> HealthCheck:
        """Perform system health check"""
        health = HealthCheck()
        
        try:
            # Component checks
            health.content_generation = self.content_pipeline is not None
            health.media_generation = self.media_pipeline is not None
            health.video_assembly = self.video_assembler is not None
            
            # Resource checks
            health.disk_space_gb = shutil.disk_usage('.').free / (1024**3)
            health.memory_usage_percent = psutil.virtual_memory().percent
            
            # GPU checks (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    health.gpu_memory_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass
            
            # Queue status
            health.pending_jobs = self.stats.pending_jobs
            health.failed_jobs_24h = sum(1 for job in self.job_history 
                                       if (job.status == ScheduleStatus.FAILED and 
                                           job.completed_at and 
                                           job.completed_at > datetime.now() - timedelta(hours=24)))
            
            # Performance
            health.success_rate_24h = self.stats.success_rate_percent
            health.average_job_time_minutes = self.stats.average_render_time / 60
            
            # Determine overall status
            if health.failed_jobs_24h > 5:
                health.overall_status = "critical"
                health.alerts.append("High failure rate in last 24 hours")
            elif health.disk_space_gb < 5:
                health.overall_status = "warning"
                health.alerts.append("Low disk space")
            elif health.memory_usage_percent > 90:
                health.overall_status = "warning"
                health.alerts.append("High memory usage")
            else:
                health.overall_status = "healthy"
            
            return health
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            health.overall_status = "critical"
            health.alerts.append(f"Health check failed: {e}")
            return health
    
    def get_queue_metrics(self) -> QueueMetrics:
        """Get queue performance metrics"""
        metrics = QueueMetrics()
        
        with self.state_lock:
            # Queue sizes
            metrics.pending_count = sum(1 for job in self.active_jobs.values() 
                                      if job.status == ScheduleStatus.PENDING)
            metrics.running_count = sum(1 for job in self.active_jobs.values() 
                                      if job.status == ScheduleStatus.RUNNING)
            
            # Today's stats
            today = datetime.now().date()
            metrics.completed_today = sum(1 for job in self.job_history 
                                        if (job.completed_at and 
                                            job.completed_at.date() == today and
                                            job.status == ScheduleStatus.COMPLETED))
            metrics.failed_today = sum(1 for job in self.job_history 
                                     if (job.completed_at and 
                                         job.completed_at.date() == today and
                                         job.status == ScheduleStatus.FAILED))
            
            # Topic distribution
            for job in self.active_jobs.values():
                category = job.topic_category
                metrics.topic_distribution[category] = metrics.topic_distribution.get(category, 0) + 1
        
        return metrics
    
    def _load_schedules(self) -> None:
        """Load schedules from disk"""
        try:
            for schedule_file in self.schedules_dir.glob("*.yaml"):
                schedule_id = schedule_file.stem
                
                with open(schedule_file) as f:
                    data = yaml.safe_load(f)
                
                schedule = ScheduleConfig(**data)
                self.schedules[schedule_id] = schedule
                
            self.logger.info(f"Loaded {len(self.schedules)} schedules")
            
        except Exception as e:
            self.logger.error(f"Failed to load schedules: {e}")
    
    def _save_schedule(self, schedule_id: str, schedule: ScheduleConfig) -> None:
        """Save schedule to disk"""
        try:
            schedule_file = self.schedules_dir / f"{schedule_id}.yaml"
            
            with open(schedule_file, 'w') as f:
                yaml.dump(schedule.dict(), f, default_flow_style=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save schedule {schedule_id}: {e}")
    
    def _load_jobs(self) -> None:
        """Load job history from disk"""
        try:
            job_file = self.jobs_dir / "job_history.json"
            
            if job_file.exists():
                with open(job_file) as f:
                    data = json.load(f)
                
                self.job_history = [VideoJob(**job_data) for job_data in data]
                self.logger.info(f"Loaded {len(self.job_history)} jobs from history")
                
        except Exception as e:
            self.logger.error(f"Failed to load job history: {e}")
    
    def _save_job(self, job: VideoJob) -> None:
        """Save job to history file"""
        try:
            # Update job in memory
            with self.state_lock:
                if job.id in self.active_jobs:
                    self.active_jobs[job.id] = job
            
            # Save complete history
            job_file = self.jobs_dir / "job_history.json"
            
            with open(job_file, 'w') as f:
                all_jobs = list(self.active_jobs.values()) + self.job_history
                json.dump([job.dict() for job in all_jobs], f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save job {job.id}: {e}")
    
    def get_stats(self) -> AutomationStats:
        """Get current automation statistics"""
        self._update_stats()
        return self.stats



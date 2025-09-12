"""
Automation Data Models

Pydantic models for the automation system.
"""

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ScheduleFrequency(str, Enum):
    """Scheduling frequency options"""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class ScheduleStatus(str, Enum):
    """Schedule execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ScheduleConfig(BaseModel):
    """Configuration for automated video generation"""
    name: str
    description: Optional[str] = None
    
    # Timing
    frequency: ScheduleFrequency = ScheduleFrequency.DAILY
    time_of_day: time = Field(default=time(8, 0))  # 8:00 AM
    timezone: str = "UTC"
    
    # Topic configuration
    topic_categories: List[str] = Field(default_factory=list)
    max_videos_per_day: int = Field(default=1, ge=1, le=10)
    
    # Quality settings
    enable_preview_mode: bool = False  # Faster rendering for testing
    quality_preset: str = "balanced"   # draft, balanced, quality
    
    # Upload settings (future implementation)
    auto_upload: bool = False
    upload_schedule_delay: int = Field(default=30, ge=0)  # minutes after generation
    
    # Notification settings
    notify_on_completion: bool = True
    notify_on_failure: bool = True
    webhook_url: Optional[str] = None
    
    # Resource management
    max_concurrent_videos: int = Field(default=1, ge=1, le=4)
    cleanup_after_days: int = Field(default=30, ge=1)
    
    # Schedule metadata
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

class VideoJob(BaseModel):
    """A single video generation job"""
    id: str
    schedule_id: str
    
    # Topic information
    topic_category: str
    topic_title: str
    subtopic: Optional[str] = None
    
    # Job status
    status: ScheduleStatus = ScheduleStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    current_step: str = "queued"
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Output paths
    video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    
    # Statistics
    render_time_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None
    video_duration_seconds: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Quality metrics
    content_quality_score: Optional[float] = None
    visual_quality_score: Optional[float] = None

class AutomationStats(BaseModel):
    """Statistics for automation system"""
    
    # Job statistics
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    pending_jobs: int = 0
    
    # Performance metrics
    average_render_time: float = 0.0
    average_file_size_mb: float = 0.0
    success_rate_percent: float = 0.0
    
    # Resource utilization
    total_disk_usage_gb: float = 0.0
    gpu_utilization_avg: float = 0.0
    memory_usage_avg_gb: float = 0.0
    
    # Recent activity
    last_job_completion: Optional[datetime] = None
    jobs_today: int = 0
    jobs_this_week: int = 0
    
    # Uptime
    system_uptime_hours: float = 0.0
    last_restart: Optional[datetime] = None

class AutomationResult(BaseModel):
    """Result of an automation operation"""
    success: bool
    message: str
    
    # Job information
    job_id: Optional[str] = None
    schedule_id: Optional[str] = None
    
    # Generated content
    video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    
    # Performance data
    execution_time_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)

class HealthCheck(BaseModel):
    """System health check result"""
    overall_status: str  # healthy, warning, critical
    
    # Component status
    content_generation: bool = True
    media_generation: bool = True
    video_assembly: bool = True
    
    # Resource checks
    disk_space_gb: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_temperature: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    
    # Dependencies
    ffmpeg_available: bool = True
    cuda_available: bool = True
    ollama_running: bool = False
    
    # Queue status
    pending_jobs: int = 0
    failed_jobs_24h: int = 0
    
    # Performance indicators
    average_job_time_minutes: float = 0.0
    success_rate_24h: float = 100.0
    
    # Alerts
    alerts: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Timestamp
    checked_at: datetime = Field(default_factory=datetime.now)

class QueueMetrics(BaseModel):
    """Queue performance metrics"""
    
    # Queue sizes
    pending_count: int = 0
    running_count: int = 0
    completed_today: int = 0
    failed_today: int = 0
    
    # Timing metrics
    average_wait_time_minutes: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    
    # Topic distribution
    topic_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Performance trends
    jobs_per_hour_avg: float = 0.0
    success_rate_trend: float = 0.0
    
    # Resource projections
    estimated_disk_usage_gb: float = 0.0
    estimated_render_time_hours: float = 0.0



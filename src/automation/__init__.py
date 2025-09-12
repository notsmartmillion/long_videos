"""
Automation System

Handles automated video generation, scheduling, and queue management.
"""

from .scheduler import VideoScheduler
from .automation_models import ScheduleConfig, AutomationResult

__all__ = [
    'VideoScheduler',
    'ScheduleConfig', 
    'AutomationResult'
]



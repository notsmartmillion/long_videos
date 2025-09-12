#!/usr/bin/env python3
"""
Logging setup for Long Video AI Automation System
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any

def setup_logging(config: 'Config') -> logging.Logger:
    """Set up logging configuration"""
    
    # Get logging config from Pydantic model
    log_config = config.logging
    level = log_config.get('level', 'INFO')
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', './logs/video_ai.log')
    max_size_mb = log_config.get('max_size_mb', 100)
    backup_count = log_config.get('backup_count', 5)
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger('video_ai')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f'video_ai.{self.__class__.__name__}')
        return self._logger

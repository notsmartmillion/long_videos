"""Topic queue management system for automated video generation"""

import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import logging


class TopicItem(BaseModel):
    """Individual topic/video to be generated"""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    title: str
    category: str  # mythology, space, history, etc.
    subtopic: Optional[str] = None
    description: Optional[str] = None
    specific_requirements: List[str] = []
    target_length_minutes: int = 120
    priority: int = 1  # 1=high, 5=low
    tags: List[str] = []
    added_date: datetime = Field(default_factory=datetime.now)
    scheduled_date: Optional[datetime] = None
    
    # Generation preferences
    style_override: Optional[str] = None
    audience_override: Optional[str] = None
    special_instructions: str = ""


class CompletedTopic(TopicItem):
    """Topic that has been completed"""
    completed_date: datetime = Field(default_factory=datetime.now)
    video_path: Optional[str] = None
    generation_time_minutes: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    stats: Dict[str, Any] = {}


class TopicQueue:
    """Manages the queue of topics to be generated"""
    
    def __init__(self, config, queue_file: str = "data/topic_queue.yaml", 
                 completed_file: str = "data/completed_topics.yaml"):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.queue_file = Path(queue_file)
        self.completed_file = Path(completed_file)
        
        # Ensure directories exist
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.completed_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.pending_topics: List[TopicItem] = []
        self.completed_topics: List[CompletedTopic] = []
        
        self._load_queues()
    
    def _load_queues(self):
        """Load existing topic queues from files"""
        try:
            # Load pending topics
            if self.queue_file.exists():
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    topics_data = data.get('topics', [])
                    self.pending_topics = [TopicItem(**item) for item in topics_data]
                    self.logger.info(f"Loaded {len(self.pending_topics)} pending topics")
            
            # Load completed topics
            if self.completed_file.exists():
                with open(self.completed_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    completed_data = data.get('completed', [])
                    self.completed_topics = [CompletedTopic(**item) for item in completed_data]
                    self.logger.info(f"Loaded {len(self.completed_topics)} completed topics")
        
        except Exception as e:
            self.logger.error(f"Failed to load topic queues: {e}")
    
    def _save_queues(self):
        """Save topic queues to files"""
        try:
            # Save pending topics
            queue_data = {
                'topics': [topic.model_dump() for topic in self.pending_topics],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                yaml.dump(queue_data, f, default_flow_style=False, sort_keys=False)
            
            # Save completed topics
            completed_data = {
                'completed': [topic.model_dump() for topic in self.completed_topics],
                'last_updated': datetime.now().isoformat(),
                'total_completed': len(self.completed_topics)
            }
            
            with open(self.completed_file, 'w', encoding='utf-8') as f:
                yaml.dump(completed_data, f, default_flow_style=False, sort_keys=False)
        
        except Exception as e:
            self.logger.error(f"Failed to save topic queues: {e}")
    
    def add_topic(self, title: str, category: str, **kwargs) -> str:
        """Add a new topic to the queue"""
        topic = TopicItem(
            title=title,
            category=category,
            **kwargs
        )
        
        self.pending_topics.append(topic)
        self._save_queues()
        
        self.logger.info(f"Added topic '{title}' to queue (ID: {topic.id})")
        return topic.id
    
    def add_topics_batch(self, topics: List[Dict[str, Any]]) -> List[str]:
        """Add multiple topics at once"""
        topic_ids = []
        
        for topic_data in topics:
            if 'title' not in topic_data or 'category' not in topic_data:
                self.logger.warning(f"Skipping invalid topic: {topic_data}")
                continue
            
            topic = TopicItem(**topic_data)
            self.pending_topics.append(topic)
            topic_ids.append(topic.id)
        
        self._save_queues()
        self.logger.info(f"Added {len(topic_ids)} topics to queue")
        return topic_ids
    
    def get_next_topic(self) -> Optional[TopicItem]:
        """Get the next topic to process (highest priority first)"""
        if not self.pending_topics:
            return None
        
        # Sort by priority (1=highest) then by added date
        self.pending_topics.sort(key=lambda x: (x.priority, x.added_date))
        
        return self.pending_topics[0]
    
    def mark_completed(self, topic_id: str, video_path: Optional[str] = None, 
                      generation_time_minutes: float = 0.0, success: bool = True,
                      error_message: Optional[str] = None, stats: Dict[str, Any] = None) -> bool:
        """Mark a topic as completed and move it to completed list"""
        
        # Find the topic in pending list
        topic_to_complete = None
        for i, topic in enumerate(self.pending_topics):
            if topic.id == topic_id:
                topic_to_complete = self.pending_topics.pop(i)
                break
        
        if not topic_to_complete:
            self.logger.warning(f"Topic ID {topic_id} not found in pending topics")
            return False
        
        # Convert to completed topic
        completed_topic = CompletedTopic(
            **topic_to_complete.model_dump(),
            video_path=video_path,
            generation_time_minutes=generation_time_minutes,
            success=success,
            error_message=error_message,
            stats=stats or {}
        )
        
        self.completed_topics.append(completed_topic)
        self._save_queues()
        
        self.logger.info(f"Marked topic '{topic_to_complete.title}' as completed")
        return True
    
    def mark_failed(self, topic_id: str, error_message: str) -> bool:
        """Mark a topic as failed"""
        return self.mark_completed(
            topic_id=topic_id,
            success=False,
            error_message=error_message
        )
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        pending_by_category = {}
        for topic in self.pending_topics:
            category = topic.category
            if category not in pending_by_category:
                pending_by_category[category] = []
            pending_by_category[category].append(topic.title)
        
        completed_by_category = {}
        for topic in self.completed_topics:
            category = topic.category
            if category not in completed_by_category:
                completed_by_category[category] = 0
            completed_by_category[category] += 1
        
        recent_completed = sorted(
            [t for t in self.completed_topics if t.success],
            key=lambda x: x.completed_date,
            reverse=True
        )[:5]
        
        return {
            'pending_count': len(self.pending_topics),
            'completed_count': len(self.completed_topics),
            'pending_by_category': pending_by_category,
            'completed_by_category': completed_by_category,
            'recent_completed': [
                {
                    'title': t.title,
                    'category': t.category,
                    'completed_date': t.completed_date.strftime('%Y-%m-%d %H:%M'),
                    'generation_time': f"{t.generation_time_minutes:.1f} minutes"
                }
                for t in recent_completed
            ],
            'failed_count': len([t for t in self.completed_topics if not t.success])
        }
    
    def list_pending_topics(self, category: Optional[str] = None) -> List[TopicItem]:
        """List all pending topics, optionally filtered by category"""
        if category:
            return [t for t in self.pending_topics if t.category == category]
        return self.pending_topics.copy()
    
    def list_completed_topics(self, category: Optional[str] = None, 
                            success_only: bool = True) -> List[CompletedTopic]:
        """List completed topics"""
        topics = self.completed_topics
        
        if success_only:
            topics = [t for t in topics if t.success]
        
        if category:
            topics = [t for t in topics if t.category == category]
        
        return topics
    
    def remove_topic(self, topic_id: str) -> bool:
        """Remove a topic from the pending queue"""
        for i, topic in enumerate(self.pending_topics):
            if topic.id == topic_id:
                removed_topic = self.pending_topics.pop(i)
                self._save_queues()
                self.logger.info(f"Removed topic '{removed_topic.title}' from queue")
                return True
        
        return False
    
    def update_topic_priority(self, topic_id: str, new_priority: int) -> bool:
        """Update topic priority"""
        for topic in self.pending_topics:
            if topic.id == topic_id:
                topic.priority = new_priority
                self._save_queues()
                return True
        return False
    
    def clear_completed(self, older_than_days: Optional[int] = None) -> int:
        """Clear old completed topics"""
        if older_than_days is None:
            count = len(self.completed_topics)
            self.completed_topics.clear()
        else:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            original_count = len(self.completed_topics)
            self.completed_topics = [
                t for t in self.completed_topics 
                if t.completed_date > cutoff_date
            ]
            count = original_count - len(self.completed_topics)
        
        self._save_queues()
        self.logger.info(f"Cleared {count} completed topics")
        return count
    
    def export_topics_to_file(self, filepath: str, include_completed: bool = True):
        """Export all topics to a file for backup"""
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'pending_topics': [t.model_dump() for t in self.pending_topics],
        }
        
        if include_completed:
            export_data['completed_topics'] = [t.model_dump() for t in self.completed_topics]
        
        export_path = Path(filepath)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.json'):
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            with open(export_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False)
    
    def import_topics_from_file(self, filepath: str) -> int:
        """Import topics from a file"""
        import_path = Path(filepath)
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {filepath}")
        
        if filepath.endswith('.json'):
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        
        imported_count = 0
        
        # Import pending topics
        if 'pending_topics' in data:
            for topic_data in data['pending_topics']:
                try:
                    topic = TopicItem(**topic_data)
                    # Generate new ID to avoid conflicts
                    topic.id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{imported_count}"
                    self.pending_topics.append(topic)
                    imported_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to import topic: {e}")
        
        self._save_queues()
        self.logger.info(f"Imported {imported_count} topics")
        return imported_count


def create_sample_topics_file(filepath: str = "data/sample_topics.yaml"):
    """Create a sample topics file for users to customize"""
    
    sample_topics = {
        'topics': [
            {
                'title': 'Zeus: King of the Olympian Gods',
                'category': 'mythology',
                'subtopic': 'Zeus',
                'description': 'Complete story of Zeus from birth to ruling Olympus',
                'tags': ['greek', 'olympian', 'thunder', 'king of gods'],
                'priority': 1
            },
            {
                'title': 'Aphrodite: Goddess of Love and Beauty', 
                'category': 'mythology',
                'subtopic': 'Aphrodite',
                'description': 'The myths and stories surrounding Aphrodite',
                'tags': ['greek', 'olympian', 'love', 'beauty'],
                'priority': 1
            },
            {
                'title': 'The Trojan War: Epic Tale of Ancient Greece',
                'category': 'mythology', 
                'subtopic': 'Trojan War',
                'description': 'Complete retelling of the Trojan War from beginning to end',
                'tags': ['greek', 'war', 'homer', 'epic'],
                'priority': 2
            },
            {
                'title': 'Europa: Jupiter\'s Mysterious Ocean Moon',
                'category': 'space',
                'subtopic': 'Europa',
                'description': 'Exploration of Jupiter\'s ice-covered moon and potential for life',
                'tags': ['jupiter', 'moon', 'ocean', 'astrobiology'],
                'priority': 1
            },
            {
                'title': 'The Fall of Rome: End of an Empire',
                'category': 'history',
                'subtopic': 'Roman Empire',
                'description': 'The complex factors that led to Rome\'s decline and fall',
                'tags': ['rome', 'empire', 'decline', 'history'],
                'priority': 2
            }
        ],
        'instructions': 'Add your topics here. Each topic should have at minimum a title and category. See examples above.',
        'categories': ['mythology', 'space', 'history', 'science', 'nature', 'culture'],
        'priority_levels': {
            1: 'High priority - generate first',
            2: 'Medium priority', 
            3: 'Normal priority',
            4: 'Low priority',
            5: 'Lowest priority - generate last'
        }
    }
    
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_topics, f, default_flow_style=False, sort_keys=False)
    
    print(f"Sample topics file created at: {filepath}")
    print("Edit this file to add your own topics!")


if __name__ == "__main__":
    # Create sample file when run directly
    create_sample_topics_file()

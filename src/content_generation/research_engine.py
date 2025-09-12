"""Research engine for gathering information on any topic"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from urllib.parse import quote
import aiohttp
import requests
from bs4 import BeautifulSoup
import feedparser

from .content_models import (
    ResearchSource, SourceType, ResearchReport, 
    ContentGenerationRequest
)
from .prompt_templates import PromptTemplates


class ResearchEngine:
    """Intelligent research engine that gathers information on any topic"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Research source configurations
        self.source_configs = {
            "wikipedia": {
                "base_url": "https://en.wikipedia.org/api/rest_v1/page/summary/",
                "search_url": "https://en.wikipedia.org/w/api.php",
                "credibility": 0.7
            },
            "britannica": {
                "search_url": "https://www.britannica.com/search?query=",
                "credibility": 0.9
            },
            "nasa": {
                "rss_feeds": [
                    "https://www.nasa.gov/news/releases/latest/index.html",
                    "https://www.nasa.gov/rss/dyn/breaking_news.rss"
                ],
                "credibility": 0.95
            },
            "scientific_journals": {
                "base_url": "https://api.crossref.org/works",
                "credibility": 0.9
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Long-Video-AI-Research/1.0 (Educational Content Generation)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def research_topic(self, request: ContentGenerationRequest) -> ResearchReport:
        """Main research method for any topic"""
        self.logger.info(f"Starting research for topic: {request.topic}")
        
        research_report = ResearchReport(
            topic=request.topic,
            subtopic=request.subtopic
        )
        
        try:
            # Multi-source research approach
            tasks = []
            
            # Wikipedia research (always available)
            tasks.append(self._research_wikipedia(request.topic, request.subtopic))
            
            # Topic-specific research based on configuration
            topic_config = self.config.get_topic_config(request.topic)
            if topic_config:
                for source in topic_config.sources:
                    if source == "NASA" and request.topic == "space":
                        tasks.append(self._research_nasa(request.topic, request.subtopic))
                    elif source == "academic_papers":
                        tasks.append(self._research_academic_sources(request.topic, request.subtopic))
                    elif source == "historical_texts" and request.topic in ["history", "mythology"]:
                        tasks.append(self._research_historical_sources(request.topic, request.subtopic))
            
            # General web research
            tasks.append(self._research_web_general(request.topic, request.subtopic))
            
            # Execute all research tasks in parallel
            research_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Compile results
            for result in research_results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Research task failed: {result}")
                    continue
                    
                if isinstance(result, list):
                    research_report.sources.extend(result)
                elif isinstance(result, ResearchSource):
                    research_report.sources.append(result)
            
            # Process and enrich the research data
            await self._enrich_research_data(research_report, request)
            
            # Calculate quality score
            research_report.research_quality_score = self._calculate_quality_score(research_report)
            
            self.logger.info(f"Research completed. Found {len(research_report.sources)} sources")
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            research_report.research_quality_score = 0.0
        
        return research_report
    
    async def _research_wikipedia(self, topic: str, subtopic: Optional[str] = None) -> List[ResearchSource]:
        """Research using Wikipedia API"""
        sources = []
        
        try:
            search_terms = [topic]
            if subtopic:
                search_terms.append(f"{topic} {subtopic}")
            
            for term in search_terms:
                # Search for articles
                search_params = {
                    'action': 'opensearch',
                    'search': term,
                    'limit': 5,
                    'format': 'json'
                }
                
                async with self.session.get(
                    self.source_configs["wikipedia"]["search_url"],
                    params=search_params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        titles = data[1] if len(data) > 1 else []
                        
                        # Get detailed info for each article
                        for title in titles[:3]:  # Limit to top 3 results
                            source = await self._get_wikipedia_article(title)
                            if source:
                                sources.append(source)
        
        except Exception as e:
            self.logger.warning(f"Wikipedia research failed: {e}")
        
        return sources
    
    async def _get_wikipedia_article(self, title: str) -> Optional[ResearchSource]:
        """Get detailed Wikipedia article information"""
        try:
            url = f"{self.source_configs['wikipedia']['base_url']}{quote(title)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return ResearchSource(
                        title=data.get('title', title),
                        url=data.get('content_urls', {}).get('desktop', {}).get('page'),
                        source_type=SourceType.ENCYCLOPEDIA,
                        credibility_score=0.7,
                        content_summary=data.get('extract', ''),
                        key_facts=[data.get('extract', '')[:200] + '...'],
                        relevance_score=0.8
                    )
        
        except Exception as e:
            self.logger.warning(f"Failed to get Wikipedia article for {title}: {e}")
        
        return None
    
    async def _research_nasa(self, topic: str, subtopic: Optional[str] = None) -> List[ResearchSource]:
        """Research using NASA APIs and feeds"""
        sources = []
        
        try:
            # NASA RSS feeds
            for feed_url in self.source_configs["nasa"]["rss_feeds"]:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:5]:  # Top 5 entries
                        if self._is_relevant_to_topic(entry.title + entry.summary, topic, subtopic):
                            source = ResearchSource(
                                title=entry.title,
                                url=entry.link,
                                source_type=SourceType.NEWS_ARTICLE,
                                credibility_score=0.95,
                                content_summary=entry.summary,
                                key_facts=[entry.summary],
                                relevance_score=self._calculate_relevance(
                                    entry.title + entry.summary, topic, subtopic
                                )
                            )
                            sources.append(source)
                
                except Exception as e:
                    self.logger.warning(f"Failed to parse NASA feed {feed_url}: {e}")
        
        except Exception as e:
            self.logger.warning(f"NASA research failed: {e}")
        
        return sources
    
    async def _research_academic_sources(self, topic: str, subtopic: Optional[str] = None) -> List[ResearchSource]:
        """Research academic papers and journals"""
        sources = []
        
        try:
            # Use CrossRef API for academic papers
            search_query = topic
            if subtopic:
                search_query += f" {subtopic}"
            
            params = {
                'query': search_query,
                'rows': 10,
                'sort': 'relevance',
                'filter': 'type:journal-article'
            }
            
            async with self.session.get(
                self.source_configs["scientific_journals"]["base_url"],
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data.get('message', {}).get('items', []):
                        if 'title' in item and 'abstract' in item:
                            source = ResearchSource(
                                title=item['title'][0] if item['title'] else 'Unknown',
                                url=item.get('URL'),
                                source_type=SourceType.ACADEMIC_PAPER,
                                credibility_score=0.9,
                                content_summary=item.get('abstract', ''),
                                key_facts=[item.get('abstract', '')[:300] + '...'],
                                citations=[f"{author.get('given', '')} {author.get('family', '')}" 
                                         for author in item.get('author', [])],
                                relevance_score=0.7
                            )
                            sources.append(source)
        
        except Exception as e:
            self.logger.warning(f"Academic source research failed: {e}")
        
        return sources
    
    async def _research_historical_sources(self, topic: str, subtopic: Optional[str] = None) -> List[ResearchSource]:
        """Research historical and cultural sources"""
        sources = []
        
        # This would integrate with historical databases, digital libraries, etc.
        # For now, we'll create placeholder functionality
        
        historical_databases = [
            {
                "name": "Perseus Digital Library",
                "url": "http://www.perseus.tufts.edu/hopper/",
                "focus": "ancient_texts"
            },
            {
                "name": "Internet Archive",
                "url": "https://archive.org/",
                "focus": "historical_documents"
            }
        ]
        
        # In a full implementation, we would query these databases
        # For now, return placeholder historical context
        if topic in ["mythology", "history"]:
            source = ResearchSource(
                title=f"Historical Context: {topic.title()}",
                source_type=SourceType.HISTORICAL_RECORD,
                credibility_score=0.8,
                content_summary=f"Historical background and context for {topic}",
                key_facts=[f"Key historical facts about {topic}"],
                relevance_score=0.9
            )
            sources.append(source)
        
        return sources
    
    async def _research_web_general(self, topic: str, subtopic: Optional[str] = None) -> List[ResearchSource]:
        """General web research using search engines"""
        sources = []
        
        # Note: In production, you might want to use search APIs like:
        # - Google Custom Search API
        # - Bing Search API
        # - DuckDuckGo API
        
        # For now, we'll simulate web research results
        search_query = topic
        if subtopic:
            search_query += f" {subtopic}"
        
        # Simulate finding relevant web sources
        simulated_sources = [
            {
                "title": f"Comprehensive Guide to {search_query}",
                "summary": f"Detailed information about {search_query} from educational resources",
                "type": SourceType.WEB_RESOURCE,
                "credibility": 0.6
            },
            {
                "title": f"Latest Research on {search_query}",
                "summary": f"Recent findings and discoveries related to {search_query}",
                "type": SourceType.WEB_RESOURCE,
                "credibility": 0.5
            }
        ]
        
        for sim_source in simulated_sources:
            source = ResearchSource(
                title=sim_source["title"],
                source_type=sim_source["type"],
                credibility_score=sim_source["credibility"],
                content_summary=sim_source["summary"],
                key_facts=[sim_source["summary"]],
                relevance_score=0.7
            )
            sources.append(source)
        
        return sources
    
    async def _enrich_research_data(self, research_report: ResearchReport, 
                                  request: ContentGenerationRequest):
        """Enrich research data with additional analysis"""
        
        # Extract key facts from all sources
        all_content = " ".join([source.content_summary for source in research_report.sources])
        
        # Extract key information (this would use NLP in a full implementation)
        research_report.key_facts = self._extract_key_facts(all_content, request.topic)
        research_report.key_figures = self._extract_key_figures(all_content, request.topic)
        research_report.locations = self._extract_locations(all_content, request.topic)
        research_report.concepts = self._extract_concepts(all_content, request.topic)
        research_report.visual_elements = self._suggest_visual_elements(request.topic, request.subtopic)
    
    def _extract_key_facts(self, content: str, topic: str) -> List[str]:
        """Extract key facts from research content"""
        # Simplified fact extraction - in production, use NLP
        facts = []
        
        # Look for common fact patterns
        sentences = content.split('.')
        for sentence in sentences[:10]:  # Top 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 50 and any(word in sentence.lower() for word in 
                                        ['discovered', 'invented', 'created', 'established', 'founded']):
                facts.append(sentence)
        
        return facts[:5]  # Return top 5 facts
    
    def _extract_key_figures(self, content: str, topic: str) -> List[Dict[str, str]]:
        """Extract key people/figures from content"""
        # Simplified name extraction
        figures = []
        
        # Common title patterns that indicate important people
        title_patterns = [
            r'(King|Queen|Emperor|President|Dr\.|Professor|Sir|Lord) (\w+ \w+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+) (?:was|is|became)',
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, content)
            for match in matches[:5]:
                if isinstance(match, tuple):
                    name = ' '.join(match)
                else:
                    name = match
                
                figures.append({
                    "name": name,
                    "role": "Key figure",
                    "relevance": "Primary"
                })
        
        return figures[:5]
    
    def _extract_locations(self, content: str, topic: str) -> List[Dict[str, str]]:
        """Extract important locations from content"""
        locations = []
        
        # Look for location patterns
        location_indicators = ['in', 'at', 'near', 'from', 'located']
        words = content.split()
        
        for i, word in enumerate(words):
            if word.lower() in location_indicators and i + 1 < len(words):
                potential_location = words[i + 1]
                if potential_location[0].isupper() and len(potential_location) > 3:
                    locations.append({
                        "name": potential_location,
                        "type": "Geographic location",
                        "significance": "Research context"
                    })
        
        return locations[:5]
    
    def _extract_concepts(self, content: str, topic: str) -> List[Dict[str, str]]:
        """Extract key concepts that need explanation"""
        concepts = []
        
        # Topic-specific concept extraction
        topic_concepts = {
            "mythology": ["pantheon", "deity", "legend", "myth", "ritual"],
            "space": ["galaxy", "planet", "star", "orbit", "cosmic"],
            "history": ["civilization", "empire", "dynasty", "era", "culture"],
            "science": ["theory", "hypothesis", "experiment", "discovery", "principle"]
        }
        
        relevant_concepts = topic_concepts.get(topic, [])
        for concept in relevant_concepts:
            if concept in content.lower():
                concepts.append({
                    "term": concept,
                    "definition": f"Key concept in {topic}",
                    "importance": "high"
                })
        
        return concepts[:5]
    
    def _suggest_visual_elements(self, topic: str, subtopic: Optional[str] = None) -> List[str]:
        """Suggest visual elements for the topic"""
        visual_suggestions = {
            "mythology": [
                "Classical paintings and sculptures",
                "Ancient temple architecture",
                "Mythological creature illustrations",
                "Historical artifacts and pottery",
                "Maps of ancient civilizations"
            ],
            "space": [
                "Telescope images of celestial objects",
                "Spacecraft and mission imagery",
                "Planetary surface photography",
                "Astronomical diagrams and charts",
                "Space exploration timeline graphics"
            ],
            "history": [
                "Historical paintings and artwork",
                "Archaeological artifacts",
                "Period-accurate reconstructions",
                "Historical maps and timelines",
                "Portrait paintings of key figures"
            ],
            "science": [
                "Scientific diagrams and charts",
                "Laboratory equipment and experiments",
                "Molecular and atomic visualizations",
                "Research photographs",
                "Data visualization graphics"
            ]
        }
        
        return visual_suggestions.get(topic, [
            "Relevant photography",
            "Illustrative diagrams",
            "Historical imagery",
            "Educational graphics",
            "Documentary-style visuals"
        ])
    
    def _is_relevant_to_topic(self, text: str, topic: str, subtopic: Optional[str] = None) -> bool:
        """Check if text content is relevant to the research topic"""
        text_lower = text.lower()
        topic_lower = topic.lower()
        
        # Basic relevance check
        if topic_lower in text_lower:
            return True
        
        if subtopic and subtopic.lower() in text_lower:
            return True
        
        # Topic-specific keywords
        topic_keywords = {
            "mythology": ["myth", "legend", "god", "goddess", "ancient", "story"],
            "space": ["space", "planet", "star", "galaxy", "cosmic", "universe"],
            "history": ["historical", "ancient", "empire", "civilization", "era"],
            "science": ["research", "study", "discovery", "scientific", "experiment"]
        }
        
        relevant_keywords = topic_keywords.get(topic, [])
        return any(keyword in text_lower for keyword in relevant_keywords)
    
    def _calculate_relevance(self, text: str, topic: str, subtopic: Optional[str] = None) -> float:
        """Calculate relevance score for text content"""
        score = 0.0
        text_lower = text.lower()
        
        # Topic mention
        if topic.lower() in text_lower:
            score += 0.5
        
        # Subtopic mention
        if subtopic and subtopic.lower() in text_lower:
            score += 0.3
        
        # Keyword relevance
        topic_keywords = {
            "mythology": ["myth", "legend", "god", "goddess", "ancient"],
            "space": ["space", "planet", "star", "galaxy", "cosmic"],
            "history": ["historical", "ancient", "empire", "civilization"],
            "science": ["research", "study", "discovery", "scientific"]
        }
        
        relevant_keywords = topic_keywords.get(topic, [])
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in text_lower)
        score += min(keyword_matches * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, research_report: ResearchReport) -> float:
        """Calculate overall quality score for research"""
        if not research_report.sources:
            return 0.0
        
        # Average credibility of sources
        avg_credibility = sum(source.credibility_score for source in research_report.sources) / len(research_report.sources)
        
        # Source diversity (different types of sources)
        source_types = set(source.source_type for source in research_report.sources)
        diversity_score = min(len(source_types) / 5, 1.0)  # Max 5 different types
        
        # Content richness
        total_content = sum(len(source.content_summary) for source in research_report.sources)
        content_score = min(total_content / 5000, 1.0)  # Target 5000 chars
        
        # Weighted average
        quality_score = (
            avg_credibility * 0.4 +
            diversity_score * 0.3 +
            content_score * 0.3
        )
        
        return round(quality_score, 2)

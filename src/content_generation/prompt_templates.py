"""Prompt templates for different types of content generation"""

from typing import Dict, List, Optional
from .content_models import ContentType, NarrativeStructure


class PromptTemplates:
    """Collection of prompt templates for various content generation tasks"""
    
    @staticmethod
    def get_research_prompt(topic: str, subtopic: Optional[str] = None, 
                          target_audience: str = "general") -> str:
        """Generate a prompt for researching a topic"""
        
        base_prompt = f"""
You are an expert researcher tasked with gathering comprehensive information about "{topic}".
{f'Specifically focusing on: {subtopic}' if subtopic else ''}

Your research should be suitable for a {target_audience} audience.

Please provide:
1. Key historical facts and timeline (if applicable)
2. Important figures/characters involved
3. Significant locations or settings
4. Core concepts that need explanation
5. Interesting stories and anecdotes
6. Modern relevance and impact
7. Common misconceptions to address
8. Visual elements that would enhance understanding

Focus on creating engaging, accurate content that would work well for a 2-hour documentary.
Ensure all information is factual and cite-worthy.
"""
        return base_prompt.strip()
    
    @staticmethod
    def get_script_generation_prompt(research_data: str, topic: str, 
                                   subtopic: Optional[str] = None,
                                   content_type: ContentType = ContentType.DOCUMENTARY,
                                   narrative_structure: NarrativeStructure = NarrativeStructure.CHRONOLOGICAL,
                                   target_length_minutes: int = 120) -> str:
        """Generate a prompt for creating the video script"""
        
        structure_guidance = {
            NarrativeStructure.CHRONOLOGICAL: "Organize the content chronologically, following events in time order",
            NarrativeStructure.THEMATIC: "Organize by themes, exploring different aspects of the topic",
            NarrativeStructure.CHARACTER_DRIVEN: "Focus on key characters/figures and their stories",
            NarrativeStructure.PROBLEM_SOLUTION: "Present challenges and their solutions",
            NarrativeStructure.JOURNEY: "Take the audience on a journey of discovery"
        }
        
        content_style = {
            ContentType.DOCUMENTARY: "documentary-style with authoritative narration",
            ContentType.EDUCATIONAL: "educational format with clear explanations",
            ContentType.NARRATIVE: "story-driven narrative format",
            ContentType.STORYTELLING: "engaging storytelling with dramatic elements",
            ContentType.CONVERSATIONAL: "conversational style with friendly, engaging tone"
        }
        
        # Calculate part-based structure (not chapters) - generic for any topic
        if target_length_minutes <= 5:
            part_count = 2
            part_length = "2-3 minutes each"
            target_words = 120
        elif target_length_minutes <= 30:
            part_count = min(6, max(3, target_length_minutes // 6))
            part_length = "4-6 minutes each"
            target_words = target_length_minutes * 150
        elif target_length_minutes <= 60:
            part_count = min(8, max(5, target_length_minutes // 8))
            part_length = "6-10 minutes each"
            target_words = target_length_minutes * 150
        else:
            part_count = min(12, max(8, target_length_minutes // 10))
            part_length = "8-12 minutes each"
            target_words = target_length_minutes * 150

        # Create the universal documentary style prompt - works for ANY topic
        if target_length_minutes <= 5:
            # Ultra-brief test mode — PURE NARRATION ONLY
            prompt = f"""
CRITICAL INSTRUCTION: Produce PURE NARRATION ONLY for a {target_length_minutes}-minute TEST video. No labels, no headers, no bullet lists, no stage directions, and absolutely NO image or visual descriptions.

TOPIC: "{topic}"{f" focusing on {subtopic}" if subtopic else ""}
TARGET: MAXIMUM 120 words total

RESEARCH DATA (summarize briefly):
{research_data[:500]}...

MANDATORY RULES:
- Do NOT include any of these: "Part 1", "Chapter", "Title:", "Opening Hook:", "Image:", "Image description:", brackets [ ... ], parentheses ( ... ) with directions, or speaker labels like "Narrator:".
- Write continuous narration as if being spoken aloud to the viewer.
- Keep it concise, cinematic, and engaging.
- End with a single-line reflective closing.

IMPLIED OUTLINE (DO NOT LABEL IN OUTPUT):
- Start with a provocative hook.
- Pivot to a second viewpoint or tension (dual perspective) in one line.
- Include one concrete evidence/reference (e.g., an era, place, or known figure) in one line.
- Connect to modern relevance in one line.
- Conclude with a philosophical reflection in one line.

OUTPUT FORMAT:
- A single block of narration prose (2–5 sentences). Nothing else.
"""
        else:
            # Full documentary template — PURE NARRATION ONLY
            prompt = f"""
You are writing a long-form documentary narration about "{topic}"{f" focusing on {subtopic}" if subtopic else ""} in the style of high-production YouTube documentaries. Create a {target_length_minutes}-minute script with a single narrator voice.

TARGET LENGTH: {target_length_minutes} minutes (~{target_words} words)

RESEARCH DATA TO INTEGRATE:
{research_data}

NARRATIVE VOICE & STYLE:
- Write like a master storyteller guiding viewers through unfolding revelations
- Use cinematic, suspenseful, and thought-provoking narration
- Blend fact-based explanation with imaginative speculation
- Balance scientific/technical clarity with philosophical reflection
- Avoid dry reporting; aim for dramatic pacing and engaging delivery
- Speak directly TO the audience, making them feel part of the journey

MANDATORY RULES (STRUCTURE IMPLIED, BUT NO LABELS):
- Do NOT include any structural labels or meta text (no "Part/Chapter" lines, no headers, no bullets).
- Do NOT include any visual directions (no "Image:" lines; no [IMAGE: ...] markers; no bracketed or parenthetical directions).
- Produce continuous narration paragraphs only.
- Open cinematically; weave mainstream understanding with speculation; connect to modern relevance; close philosophically.

IMPLIED OUTLINE (DO NOT LABEL IN OUTPUT):
- Paragraph 1: Cinematic hook that stakes the core idea in vivid, concrete language.
- Paragraph 2: Establish context and introduce a dual perspective (conventional view vs deeper possibility).
- Paragraph 3: Evidence integration (dates, places, discoveries, expert views) that grounds the narrative.
- Paragraph 4: Speculation moment—pose a “What if…?” and explore a plausible alternative, then temper with known facts.
- Paragraph 5: Counterpoint or tension—recognize limits or an opposing interpretation to keep credibility.
- Paragraph 6: Modern relevance—show how echoes of the topic persist today (culture, tech, language, ethics).
- Paragraph 7: Synthesis—bridge the evidence and speculation into a coherent takeaway.
- Paragraph 8–9: Philosophical reflection—zoom out to human meaning; end with a resonant line.

STYLE CUES (APPLY, DO NOT PRINT):
- Use short-to-medium sentences with occasional rhetorical questions for rhythm.
- Prefer concrete nouns and active verbs; avoid listy exposition.
- Maintain measured suspense; vary cadence; avoid fluff.

OUTPUT FORMAT:
- Title line (one line, no "Title:" label).
- 5–9 paragraphs of pure narration prose.
- Final single-paragraph philosophical reflection.

CRITICAL VOICE REQUIREMENTS:
- Mystery and revelation pacing: setup → evidence → speculation → deeper truth
- Balance wonder with rigor; avoid filler and jargon
- Aim for ~even paragraph distribution, totaling ~{target_words} words
"""
        return prompt.strip()
    
    @staticmethod
    def get_image_prompt_generation(script_text: str, topic: str, 
                                   visual_style: str) -> str:
        """Generate prompts for creating image prompts from script"""
        
        prompt = f"""
You are a visual director for documentaries. Analyze this script excerpt and create detailed image prompts.

SCRIPT EXCERPT:
{script_text}

TOPIC: {topic}
VISUAL STYLE: {visual_style}

For each [IMAGE: ...] marker in the script, create a detailed prompt that includes:
- Main subject/scene
- Composition and framing
- Lighting and mood
- Color palette
- Art style consistent with {visual_style}
- Any specific details mentioned in the script

FORMAT each image prompt as:
TIMESTAMP: [time in script]
PROMPT: [detailed image generation prompt]
STYLE: [specific style modifiers]
CONTEXT: [what's being discussed]

Make each prompt vivid and specific enough for AI image generation.
"""
        return prompt.strip()
    
    @staticmethod
    def get_chapter_breakdown_prompt(topic: str, research_data: str, 
                                   target_chapters: int = 8) -> str:
        """Generate a prompt for breaking content into chapters"""
        
        prompt = f"""
You are a documentary producer planning the structure for a film about "{topic}".

RESEARCH DATA:
{research_data}

Create a compelling {target_chapters}-chapter structure that:
1. Has a logical flow and progression
2. Each chapter is roughly equal in length (15-20 minutes)
3. Builds audience engagement throughout
4. Includes natural cliffhangers/transitions
5. Covers all major aspects of the topic

For each chapter, provide:
- Chapter number and title
- Key points to cover
- Estimated duration
- Transition to next chapter

FORMAT:
Chapter 1: [Title]
- Key points: [list]
- Duration: [minutes]
- Transition: [how it leads to next chapter]

[Continue for all chapters...]
"""
        return prompt.strip()
    
    @staticmethod
    def get_title_generation_prompt(topic: str, subtopic: Optional[str] = None,
                                  script_summary: str = "") -> str:
        """Generate prompts for creating compelling titles"""
        
        prompt = f"""
You are a YouTube content strategist. Create compelling titles for a 2-hour documentary about "{topic}"{f" focusing on {subtopic}" if subtopic else ""}.

{f"SCRIPT SUMMARY: {script_summary}" if script_summary else ""}

Generate 10 different title options that are:
1. SEO-friendly and searchable
2. Compelling and click-worthy
3. Accurate to the content
4. Appropriate for a 2+ hour documentary
5. Appeal to both casual viewers and enthusiasts

Include a mix of:
- Straightforward descriptive titles
- Intriguing question-based titles  
- Epic/dramatic titles
- Educational/informative titles

FORMAT:
1. [Title] - [Brief explanation of approach]
2. [Title] - [Brief explanation of approach]
...
"""
        return prompt.strip()
    
    @staticmethod
    def get_description_generation_prompt(title: str, script_summary: str,
                                        chapters: List[str]) -> str:
        """Generate YouTube description"""
        
        chapters_list = "\n".join([f"{i+1}. {chapter}" for i, chapter in enumerate(chapters)])
        
        prompt = f"""
Create a compelling YouTube description for this documentary:

TITLE: {title}
SCRIPT SUMMARY: {script_summary}
CHAPTERS:
{chapters_list}

The description should:
1. Hook viewers in the first 2 lines
2. Provide a compelling overview
3. Include chapter timestamps (estimate times)
4. Include relevant hashtags
5. Be SEO optimized
6. Invite engagement (likes, comments, subscribes)
7. Include disclaimer about AI-generated content

Keep it informative but engaging, suitable for a long-form documentary.
"""
        return prompt.strip()
    
    @staticmethod
    def get_fact_checking_prompt(content: str, topic: str) -> str:
        """Generate prompt for fact-checking content"""
        
        prompt = f"""
You are a fact-checker reviewing content about "{topic}".

CONTENT TO REVIEW:
{content}

Please:
1. Identify any factual claims
2. Flag anything that seems questionable or unverified
3. Suggest areas that need citations
4. Note any potential biases or one-sided perspectives
5. Recommend additional context that should be included
6. Identify claims that would benefit from multiple sources

Provide your feedback in a structured format with specific line references where possible.
"""
        return prompt.strip()
    
    @staticmethod
    def get_content_improvement_prompt(script: str, feedback: str) -> str:
        """Generate prompt for improving content based on feedback"""
        
        prompt = f"""
You are a script editor improving a documentary script.

ORIGINAL SCRIPT:
{script}

FEEDBACK TO ADDRESS:
{feedback}

Please revise the script to:
1. Address all feedback points
2. Maintain the original structure and flow
3. Improve clarity and engagement
4. Ensure factual accuracy
5. Enhance visual storytelling elements

Provide the revised script with [REVISED] markers showing what changed.
"""
        return prompt.strip()
    
    @staticmethod
    def get_style_adaptation_prompt(content: str, original_style: str, 
                                  target_style: str) -> str:
        """Generate prompt for adapting content to different styles"""
        
        prompt = f"""
Adapt this content from {original_style} style to {target_style} style.

ORIGINAL CONTENT:
{content}

Maintain all factual information while adjusting:
- Tone and voice
- Pacing and rhythm
- Language complexity
- Narrative approach
- Engagement techniques

Ensure the adapted version is authentic to the {target_style} format while keeping the core information intact.
"""
        return prompt.strip()

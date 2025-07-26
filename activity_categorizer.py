"""
activity_categorizer.py
─────────────────────
Categorizes user activities based on screen captures (text + images).
Creates a structured list of user actions that can be analyzed for task inference.

Features:
• Analyzes OCR text and images to determine user activity
• Categorizes activities (work, social, entertainment, etc.)
• Tracks time spent on different activities
• Stores activity history for pattern analysis
"""

import json
import os
import glob
import datetime as dt
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import openai

import dotenv
dotenv.load_dotenv()

# ─── Activity Categories ──────────────────────────────────────────────────────
class ActivityCategory(Enum):
    WORK = "work"
    SOCIAL_MEDIA = "social_media"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    UNKNOWN = "unknown"

# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class UserActivity:
    timestamp: str
    category: str
    subcategory: str
    description: str
    confidence: float
    text_content: str
    image_path: Optional[str] = None
    duration_seconds: int = 0

# ─── Config ───────────────────────────────────────────────────────────────────
CAP_DIR = Path("captures")
ACTIVITIES_JSON = Path("user_activities.json")

# ─── Activity Detection Patterns ──────────────────────────────────────────────
ACTIVITY_PATTERNS = {
    ActivityCategory.WORK: [
        "linkedin", "github", "stack overflow", "jira", "confluence", "slack",
        "zoom", "teams", "meet", "code", "programming", "development",
        "excel", "sheets", "powerpoint", "presentation", "document"
    ],
    ActivityCategory.SOCIAL_MEDIA: [
        "facebook", "instagram", "twitter", "x.com", "tiktok", "snapchat",
        "reddit", "discord", "whatsapp", "telegram", "youtube"
    ],
    ActivityCategory.ENTERTAINMENT: [
        "netflix", "spotify", "hulu", "disney+", "prime video", "gaming",
        "game", "stream", "movie", "music", "podcast"
    ],
    ActivityCategory.SHOPPING: [
        "amazon", "ebay", "etsy", "shopify", "paypal", "stripe",
        "checkout", "cart", "buy", "purchase", "order"
    ],
    ActivityCategory.LEARNING: [
        "coursera", "udemy", "khan academy", "edx", "wikipedia",
        "tutorial", "course", "learn", "study", "research"
    ],
    ActivityCategory.COMMUNICATION: [
        "gmail", "outlook", "mail", "email", "message", "chat",
        "skype", "facetime", "call", "meeting"
    ],
    ActivityCategory.PRODUCTIVITY: [
        "notion", "trello", "asana", "todo", "task", "calendar",
        "notes", "planning", "organize", "schedule"
    ]
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_latest_captures() -> Dict[str, Optional[str]]:
    """Load latest text and image captures"""
    txt_files = sorted(CAP_DIR.glob("*.txt"), reverse=True)
    # Only look for image files, not JSON files
    image_files = sorted(CAP_DIR.glob("*.png"), reverse=True) + sorted(CAP_DIR.glob("*.jpg"), reverse=True) + sorted(CAP_DIR.glob("*.jpeg"), reverse=True)
    
    result = {"text": None, "image_path": None, "image_description": None}
    
    if txt_files:
        result["text"] = txt_files[0].read_text()
    
    if image_files:
        result["image_path"] = str(image_files[0])
        result["image_description"] = analyze_image(image_files[0])
    
    return result

def analyze_image(image_path: Path) -> str:
    """Analyze image using OpenAI Vision API with rich descriptions"""
    try:
        client = openai.OpenAI()
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this screenshot and provide a rich, detailed description of what the user is doing. 

Focus on:
1. **Specific Context**: If on LinkedIn, describe the profile (job title, company, industry, experience level)
2. **Search Intent**: If searching, describe what they're looking for (apartment features, job requirements, etc.)
3. **Content Details**: If reading/watching, describe the topic, source, and relevance
4. **Work Context**: If coding/working, describe the project, technology, or task
5. **Shopping Intent**: If shopping, describe the product category, price range, or specific needs

Examples:
- "Viewing LinkedIn profile of Senior Software Engineer at Google with 8+ years experience in AI/ML"
- "Searching for 2-bedroom apartments in downtown area with parking and gym access"
- "Reading article about React performance optimization on Medium"
- "Working on Python script for data analysis using pandas and matplotlib"
- "Browsing Amazon for wireless headphones under $100 with noise cancellation"

Be specific and actionable - this helps with task inference."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return f"Image analysis failed: {str(e)}"

def categorize_activity(text_content: str, image_description: str = "") -> Tuple[str, str, float]:
    """Categorize activity using OpenAI and pattern matching"""
    try:
        client = openai.OpenAI()
        
        # Combine text and image analysis
        combined_content = f"Text: {text_content}\nImage: {image_description}"
        
        prompt = f"""
Analyze this user activity and categorize it with rich context:

{combined_content}

Categorize into one of these categories:
- work: Professional work, coding, business tasks
- social_media: Social networking, social media platforms
- entertainment: Movies, games, music, streaming
- shopping: Online shopping, e-commerce
- learning: Education, tutorials, research
- communication: Email, messaging, calls
- productivity: Task management, planning, organization

Provide a rich, detailed description that includes:
1. **Specific Context**: What exactly they're doing (e.g., "Viewing LinkedIn profile of Senior Data Scientist at Netflix")
2. **Intent**: What they might be trying to achieve (e.g., "Likely networking or job searching in tech industry")
3. **Relevant Details**: Key information that could inform task suggestions (e.g., "Profile shows 5+ years experience in machine learning")

Respond with JSON format:
{{
    "category": "category_name",
    "subcategory": "specific_activity", 
    "description": "rich detailed description with context and intent",
    "confidence": 0.95
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        # Get the response content and handle potential JSON parsing issues
        response_content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response (in case it's wrapped in other text)
        if response_content.startswith('```json'):
            response_content = response_content[7:]
        if response_content.endswith('```'):
            response_content = response_content[:-3]
        response_content = response_content.strip()
        
        # Parse JSON with better error handling
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            print(f"Response content: {response_content}")
            # Fallback to pattern matching
            return fallback_categorization(text_content)
        
        # Validate required fields
        required_fields = ["category", "subcategory", "confidence"]
        for field in required_fields:
            if field not in result:
                print(f"Missing field '{field}' in response")
                return fallback_categorization(text_content)
        
        return result["category"], result["subcategory"], result["confidence"]
        
    except Exception as e:
        print(f"Error categorizing activity: {e}")
        # Fallback to pattern matching
        return fallback_categorization(text_content)

def fallback_categorization(text_content: str) -> Tuple[str, str, float]:
    """Fallback categorization using pattern matching"""
    text_lower = text_content.lower()
    
    for category, patterns in ACTIVITY_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return category.value, pattern, 0.7
    
    return ActivityCategory.UNKNOWN.value, "unknown", 0.5

def load_activities() -> List[Dict]:
    """Load existing activities"""
    if ACTIVITIES_JSON.exists():
        with open(ACTIVITIES_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def save_activities(activities: List[Dict]):
    """Save activities to file"""
    with open(ACTIVITIES_JSON, "w", encoding="utf-8") as fh:
        json.dump(activities, fh, indent=2)

def calculate_duration(activities: List[Dict]) -> int:
    """Calculate duration since last activity (assume 1 minute intervals)"""
    if len(activities) < 2:
        return 60  # Default 1 minute
    
    # Calculate time difference between last two activities
    last_time = dt.datetime.fromisoformat(activities[-1]["timestamp"])
    prev_time = dt.datetime.fromisoformat(activities[-2]["timestamp"])
    duration = (last_time - prev_time).total_seconds()
    
    return max(60, int(duration))  # Minimum 1 minute

# ─── Main Processing ──────────────────────────────────────────────────────────
def process_activity() -> Optional[UserActivity]:
    """Process latest capture and categorize activity"""
    captures = load_latest_captures()
    
    if not captures["text"] and not captures["image_description"]:
        return None
    
    # Combine text and image analysis
    text_content = captures["text"] or ""
    image_description = captures["image_description"] or ""
    
    # Categorize the activity
    category, subcategory, confidence = categorize_activity(text_content, image_description)
    
    # Create activity record
    activity = UserActivity(
        timestamp=dt.datetime.utcnow().isoformat(),
        category=category,
        subcategory=subcategory,
        description=f"{text_content[:100]}..." if len(text_content) > 100 else text_content,
        confidence=confidence,
        text_content=text_content,
        image_path=captures["image_path"]
    )
    
    # Load existing activities and add new one
    activities = load_activities()
    activities.append(asdict(activity))
    
    # Calculate duration for the new activity
    if len(activities) > 1:
        activities[-1]["duration_seconds"] = calculate_duration(activities)
    
    # Keep only last 1000 activities to prevent file from growing too large
    if len(activities) > 1000:
        activities = activities[-1000:]
    
    save_activities(activities)
    return activity

def get_activity_summary(hours: int = 24) -> Dict:
    """Get summary of activities in the last N hours"""
    activities = load_activities()
    
    if not activities:
        return {"total_activities": 0, "categories": {}}
    
    # Filter activities from last N hours
    cutoff_time = dt.datetime.utcnow() - dt.timedelta(hours=hours)
    recent_activities = [
        act for act in activities 
        if dt.datetime.fromisoformat(act["timestamp"]) > cutoff_time
    ]
    
    # Group by category
    category_summary = {}
    for activity in recent_activities:
        category = activity["category"]
        if category not in category_summary:
            category_summary[category] = {
                "count": 0,
                "total_duration": 0,
                "subcategories": {}
            }
        
        category_summary[category]["count"] += 1
        category_summary[category]["total_duration"] += activity.get("duration_seconds", 60)
        
        # Track subcategories
        subcategory = activity["subcategory"]
        if subcategory not in category_summary[category]["subcategories"]:
            category_summary[category]["subcategories"][subcategory] = 0
        category_summary[category]["subcategories"][subcategory] += 1
    
    return {
        "total_activities": len(recent_activities),
        "time_period_hours": hours,
        "categories": category_summary
    }

def main():
    """Main function to process latest activity"""
    print(f"Processing activity at {dt.datetime.now().isoformat()}")
    
    activity = process_activity()
    
    if activity:
        print(f"✅ Categorized activity: {activity.category} - {activity.subcategory}")
        print(f"   Description: {activity.description}")
        print(f"   Confidence: {activity.confidence:.2f}")
        
        # Show recent summary
        summary = get_activity_summary(1)  # Last hour
        print(f"\n📊 Last hour summary: {summary['total_activities']} activities")
        
        for category, data in summary["categories"].items():
            duration_minutes = data["total_duration"] // 60
            print(f"   {category}: {data['count']} activities ({duration_minutes} min)")
    else:
        print("❌ No new activity found")

if __name__ == "__main__":
    main()
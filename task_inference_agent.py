"""
task_inference_agent.py
─────────────────────
Analyzes user activity patterns and infers potential todo items.
Uses AI to suggest tasks based on behavior patterns, time spent, and activity frequency.

Features:
• Analyzes activity patterns over time
• Identifies potential tasks from behavior
• Suggests productivity improvements
• Learns from user patterns to make better suggestions
"""

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import openai

import dotenv
dotenv.load_dotenv()

# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class InferredTask:
    task_id: str
    title: str
    description: str
    category: str
    priority: str  # "high", "medium", "low"
    confidence: float
    reasoning: str
    suggested_deadline: Optional[str] = None
    created_at: str = ""
    status: str = "suggested"

# ─── Config ───────────────────────────────────────────────────────────────────
ACTIVITIES_JSON = Path("user_activities.json")
TASKS_JSON = Path("tasks.json")
INFERRED_TASKS_JSON = Path("inferred_tasks.json")

# ─── Analysis Patterns ────────────────────────────────────────────────────────
TASK_INFERENCE_PATTERNS = {
    "work": {
        "linkedin": [
            "Connect with [specific person] to discuss [their expertise]",
            "Follow up on job application at [company]",
            "Update LinkedIn profile with recent [project/achievement]",
            "Share professional content about [topic from their activity]",
            "Schedule informational interview with [person/role]"
        ],
        "github": [
            "Review pull requests for [specific project]",
            "Update documentation for [feature/project]",
            "Fix bugs in [repository name]",
            "Contribute to [specific open source project]",
            "Create issue for [feature request]"
        ],
        "jira": [
            "Update sprint status for [project]",
            "Review tasks for [specific sprint]",
            "Plan next sprint for [project]",
            "Update user stories for [feature]",
            "Schedule sprint planning meeting"
        ],
        "coding": [
            "Complete [specific feature] implementation",
            "Optimize [specific function/algorithm]",
            "Write tests for [specific module]",
            "Refactor [specific code section]",
            "Deploy [specific project] to production"
        ]
    },
    "social_media": {
        "facebook": [
            "Respond to messages from [specific person]",
            "Schedule social media posts about [topic]",
            "Engage with community in [specific group]",
            "Review and update privacy settings",
            "Plan content calendar for [month/week]"
        ],
        "instagram": [
            "Post new content about [specific topic]",
            "Engage with followers on [specific post]",
            "Update bio/profile with [new information]",
            "Plan content calendar for [specific theme]",
            "Research trending hashtags for [niche]"
        ]
    },
    "learning": {
        "coursera": [
            "Complete [specific course] modules",
            "Submit assignments for [course name]",
            "Review course materials for [topic]",
            "Take final exam for [course]",
            "Apply [learned concept] to current project"
        ],
        "udemy": [
            "Finish video tutorials for [specific course]",
            "Complete practice exercises for [topic]",
            "Review course notes for [concept]",
            "Build project from [course name]",
            "Implement [learned technique] in work"
        ],
        "reading": [
            "Apply [specific technique] from article to project",
            "Research more about [topic from article]",
            "Share insights from [article] with team",
            "Create summary of [article] for reference",
            "Follow up on [topic] mentioned in article"
        ]
    },
    "shopping": {
        "apartment": [
            "Schedule viewing for [specific apartment]",
            "Contact landlord about [specific unit]",
            "Compare [specific apartments] side by side",
            "Apply for [specific apartment]",
            "Research neighborhood for [specific area]"
        ],
        "job_search": [
            "Apply for [specific job] at [company]",
            "Follow up on application at [company]",
            "Prepare for interview at [company]",
            "Research [company] culture and values",
            "Update resume for [specific role]"
        ],
        "product": [
            "Compare [specific products] for purchase",
            "Read reviews for [specific product]",
            "Check prices for [product] across retailers",
            "Research alternatives to [product]",
            "Make purchase decision for [product]"
        ]
    }
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_activities() -> List[Dict]:
    """Load user activities"""
    if ACTIVITIES_JSON.exists():
        with open(ACTIVITIES_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def load_existing_tasks() -> List[Dict]:
    """Load existing tasks"""
    if TASKS_JSON.exists():
        with open(TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"tasks": []}

def load_inferred_tasks() -> List[Dict]:
    """Load previously inferred tasks"""
    if INFERRED_TASKS_JSON.exists():
        with open(INFERRED_TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def save_inferred_tasks(tasks: List[Dict]):
    """Save inferred tasks"""
    with open(INFERRED_TASKS_JSON, "w", encoding="utf-8") as fh:
        json.dump(tasks, fh, indent=2)

def analyze_activity_patterns(hours: int = 24) -> Dict:
    """Analyze activity patterns for task inference"""
    activities = load_activities()
    
    if not activities:
        return {}
    
    # Filter recent activities
    cutoff_time = dt.datetime.utcnow() - dt.timedelta(hours=hours)
    recent_activities = [
        act for act in activities 
        if dt.datetime.fromisoformat(act["timestamp"]) > cutoff_time
    ]
    
    # Analyze patterns
    patterns = {
        "total_time": 0,
        "category_breakdown": defaultdict(int),
        "subcategory_breakdown": defaultdict(int),
        "frequent_activities": [],
        "time_distribution": defaultdict(int),
        "activity_sequences": []
    }
    
    for activity in recent_activities:
        # Validate activity has required fields
        if not isinstance(activity, dict):
            print(f"Warning: Skipping invalid activity (not a dict): {activity}")
            continue
            
        if "category" not in activity or "subcategory" not in activity:
            print(f"Warning: Skipping activity missing required fields: {activity}")
            continue
        
        duration = activity.get("duration_seconds", 60)
        patterns["total_time"] += duration
        
        category = activity["category"]
        subcategory = activity["subcategory"]
        
        patterns["category_breakdown"][category] += duration
        patterns["subcategory_breakdown"][subcategory] += duration
        
        # Track activity sequences
        if len(patterns["activity_sequences"]) > 0:
            last_activity = patterns["activity_sequences"][-1]
            if last_activity.get("category") != category:
                patterns["activity_sequences"].append({
                    "from": last_activity.get("category", "unknown"),
                    "to": category,
                    "subcategory": subcategory
                })
        else:
            patterns["activity_sequences"].append({
                "category": category,
                "subcategory": subcategory
            })
    
    # Find frequent activities (more than 30 minutes)
    frequent_activities = [
        (subcat, duration) for subcat, duration in patterns["subcategory_breakdown"].items()
        if duration > 1800  # 30 minutes
    ]
    patterns["frequent_activities"] = sorted(frequent_activities, key=lambda x: x[1], reverse=True)
    
    return patterns

def infer_tasks_with_ai(patterns: Dict) -> List[InferredTask]:
    """Use AI to infer tasks from activity patterns"""
    try:
        client = openai.OpenAI()
        
        # Prepare pattern summary for AI
        pattern_summary = f"""
Activity Analysis Summary:
- Total time: {patterns['total_time'] // 60} minutes
- Most time spent on: {dict(patterns['category_breakdown'])}
- Frequent activities: {patterns['frequent_activities']}
- Activity sequences: {patterns['activity_sequences'][-5:] if patterns['activity_sequences'] else []}
"""
        
        prompt = f"""
Based on this user activity analysis, suggest 3-5 specific, actionable tasks that the user might want to add to their todo list.

{pattern_summary}

Consider the rich context from activity descriptions:
1. **LinkedIn Activity**: If viewing specific profiles → suggest personalized networking tasks
2. **Search Activity**: If searching for apartments/jobs → suggest follow-up actions based on criteria
3. **Learning Activity**: If reading specific topics → suggest related learning tasks
4. **Work Activity**: If working on specific projects → suggest related project tasks
5. **Shopping Activity**: If browsing specific products → suggest research or purchase tasks

Use the detailed activity descriptions to create highly specific tasks:
- "Viewing Senior Data Scientist profile" → "Connect with [Name] to discuss ML opportunities"
- "Searching for 2BR apartments downtown" → "Schedule apartment viewing for [specific complex]"
- "Reading React optimization article" → "Apply performance techniques to current project"
- "Working on Python data analysis" → "Complete pandas optimization for [specific dataset]"

For each task, provide:
- A clear, specific title based on the activity context
- A detailed description with actionable steps
- Priority (high/medium/low) based on activity frequency and context
- Reasoning that references the specific activity details

Respond in JSON format:
{{
    "tasks": [
        {{
            "title": "Specific task based on activity context",
            "description": "Detailed actionable steps",
            "priority": "high|medium|low",
            "reasoning": "Why this task is suggested based on specific activity details"
        }}
    ]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
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
            return infer_tasks_with_patterns(patterns)
        
        # Validate required fields
        if "tasks" not in result:
            print("Missing 'tasks' field in response")
            return infer_tasks_with_patterns(patterns)
        
        # Convert to InferredTask objects
        inferred_tasks = []
        existing_inferred = load_inferred_tasks()
        
        for i, task_data in enumerate(result["tasks"]):
            # Validate task data
            required_task_fields = ["title", "description", "priority", "reasoning"]
            for field in required_task_fields:
                if field not in task_data:
                    print(f"Missing field '{field}' in task data")
                    continue
            
            task_id = f"inferred_{len(existing_inferred) + i + 1}"
            
            task = InferredTask(
                task_id=task_id,
                title=task_data["title"],
                description=task_data["description"],
                category="inferred",
                priority=task_data["priority"],
                confidence=0.8,  # Default confidence for AI suggestions
                reasoning=task_data["reasoning"],
                created_at=dt.datetime.utcnow().isoformat()
            )
            
            inferred_tasks.append(task)
        
        return inferred_tasks
        
    except Exception as e:
        print(f"Error inferring tasks with AI: {e}")
        return infer_tasks_with_patterns(patterns)

def infer_tasks_with_patterns(patterns: Dict) -> List[InferredTask]:
    """Fallback task inference using pattern matching"""
    inferred_tasks = []
    existing_inferred = load_inferred_tasks()
    
    # Check for frequent activities and suggest tasks
    for subcategory, duration in patterns["frequent_activities"]:
        if duration > 3600:  # More than 1 hour
            # Look for matching patterns
            for category, subcategories in TASK_INFERENCE_PATTERNS.items():
                if subcategory in subcategories:
                    for suggested_task in subcategories[subcategory]:
                        task_id = f"inferred_{len(existing_inferred) + len(inferred_tasks) + 1}"
                        
                        task = InferredTask(
                            task_id=task_id,
                            title=suggested_task,
                            description=f"Suggested based on frequent {subcategory} activity",
                            category=category,
                            priority="medium",
                            confidence=0.7,
                            reasoning=f"Spent {duration // 60} minutes on {subcategory}",
                            created_at=dt.datetime.utcnow().isoformat()
                        )
                        
                        inferred_tasks.append(task)
                        break  # Only suggest one task per subcategory
    
    return inferred_tasks

def filter_duplicate_tasks(new_tasks: List[InferredTask], existing_tasks: List[Dict]) -> List[InferredTask]:
    """Filter out tasks that are too similar to existing ones"""
    filtered_tasks = []
    
    for new_task in new_tasks:
        is_duplicate = False
        
        # Check against existing tasks
        for existing_task in existing_tasks:
            if existing_task.get("intent", "").lower() in new_task.title.lower():
                is_duplicate = True
                break
        
        # Check against existing inferred tasks
        existing_inferred = load_inferred_tasks()
        for existing_inferred_task in existing_inferred:
            if existing_inferred_task.get("title", "").lower() in new_task.title.lower():
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_tasks.append(new_task)
    
    return filtered_tasks

def suggest_productivity_improvements(patterns: Dict) -> List[str]:
    """Suggest productivity improvements based on patterns"""
    suggestions = []
    
    total_time = patterns["total_time"]
    work_time = patterns["category_breakdown"].get("work", 0)
    social_time = patterns["category_breakdown"].get("social_media", 0)
    entertainment_time = patterns["category_breakdown"].get("entertainment", 0)
    
    # Calculate percentages
    if total_time > 0:
        work_percentage = (work_time / total_time) * 100
        social_percentage = (social_time / total_time) * 100
        entertainment_percentage = (entertainment_time / total_time) * 100
        
        if social_percentage > 30:
            suggestions.append("Consider setting time limits for social media to improve productivity")
        
        if entertainment_percentage > 40:
            suggestions.append("High entertainment time detected - consider scheduling focused work blocks")
        
        if work_percentage < 20:
            suggestions.append("Low work time detected - consider prioritizing work tasks")
    
    return suggestions

# ─── Main Processing ──────────────────────────────────────────────────────────
def analyze_and_infer_tasks() -> Tuple[List[InferredTask], List[str]]:
    """Main function to analyze patterns and infer tasks"""
    try:
        print(f"🔍 Analyzing activity patterns at {dt.datetime.now().isoformat()}")
        
        # Analyze patterns
        patterns = analyze_activity_patterns(24)  # Last 24 hours
        
        if not patterns:
            print("❌ No activity data found for analysis")
            return [], []
        
        print(f"📊 Found {patterns['total_time'] // 60} minutes of activity")
        
        # Infer tasks
        inferred_tasks = infer_tasks_with_ai(patterns)
        
        # Filter duplicates
        existing_tasks = load_existing_tasks()
        filtered_tasks = filter_duplicate_tasks(inferred_tasks, existing_tasks["tasks"])
        
        # Get productivity suggestions
        productivity_suggestions = suggest_productivity_improvements(patterns)
        
        # Save new inferred tasks
        if filtered_tasks:
            existing_inferred = load_inferred_tasks()
            existing_inferred.extend([asdict(task) for task in filtered_tasks])
            save_inferred_tasks(existing_inferred)
        
        return filtered_tasks, productivity_suggestions
        
    except KeyError as e:
        print(f"❌ KeyError in task inference: {e}")
        print(f"   This usually means an activity record is missing the '{e}' field")
        return [], []
    except Exception as e:
        print(f"❌ Unexpected error in task inference: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def display_analysis_results(tasks: List[InferredTask], suggestions: List[str]):
    """Display analysis results in a user-friendly format"""
    print(f"\n🎯 Task Suggestions ({len(tasks)} new tasks):")
    
    for task in tasks:
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
        print(f"   {priority_emoji} {task.title}")
        print(f"      📝 {task.description}")
        print(f"      💭 {task.reasoning}")
        print()
    
    if suggestions:
        print(f"💡 Productivity Suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        print()

def main():
    """Main function"""
    tasks, suggestions = analyze_and_infer_tasks()
    display_analysis_results(tasks, suggestions)

if __name__ == "__main__":
    main()
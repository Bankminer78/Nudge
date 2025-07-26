"""
llama_task_inference_agent.py
───────────────────────────
LlamaIndex-based task inference agent with advanced pattern analysis.
Uses vector storage and retrieval for better task suggestion and productivity insights.

Features:
• Vector-based activity pattern analysis
• Advanced task inference using LlamaIndex
• Semantic similarity for task matching
• Rich context extraction for task generation
"""

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import openai
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
from pydantic import BaseModel

import dotenv
dotenv.load_dotenv()

# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class InferredTask:
    task_id: str
    title: str
    description: str
    category: str
    priority: str
    confidence: float
    reasoning: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None

class TaskDocument(BaseModel):
    """Pydantic model for task documents"""
    task_id: str
    title: str
    description: str
    category: str
    priority: str
    confidence: float
    reasoning: str
    created_at: str
    metadata: Dict[str, Any] = {}

# ─── Config ───────────────────────────────────────────────────────────────────
ACTIVITIES_JSON = Path("user_activities.json")
TASKS_JSON = Path("tasks.json")
INFERRED_TASKS_JSON = Path("inferred_tasks.json")
VECTOR_DB_DIR = Path("vector_db")
CHROMA_DB_DIR = Path("chroma_db")

# ─── LlamaIndex Setup ─────────────────────────────────────────────────────────
def setup_llama_index():
    """Initialize LlamaIndex with OpenAI components"""
    try:
        # Initialize OpenAI components
        llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        return llm, embed_model
    except Exception as e:
        print(f"Error setting up LlamaIndex: {e}")
        return None, None

def get_task_vector_store():
    """Get or create vector store for tasks"""
    try:
        # Create directories if they don't exist
        VECTOR_DB_DIR.mkdir(exist_ok=True)
        CHROMA_DB_DIR.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        chroma_collection = chroma_client.get_or_create_collection("tasks")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        return vector_store
    except Exception as e:
        print(f"Error creating task vector store: {e}")
        return None

def get_task_index():
    """Get or create task index"""
    try:
        vector_store = get_task_vector_store()
        if not vector_store:
            return None
        
        # Try to load existing index
        if VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir()):
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = load_index_from_storage(storage_context)
        else:
            # Create new index
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex([], storage_context=storage_context)
        
        return index
    except Exception as e:
        print(f"Error getting task index: {e}")
        return None

# ─── Task Inference Patterns ──────────────────────────────────────────────────
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

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_activities() -> List[Dict]:
    """Load user activities"""
    if ACTIVITIES_JSON.exists():
        with open(ACTIVITIES_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def load_existing_tasks() -> Dict:
    """Load existing tasks"""
    if TASKS_JSON.exists():
        with open(TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"tasks": []}

def load_inferred_tasks() -> List[Dict]:
    """Load inferred tasks"""
    if INFERRED_TASKS_JSON.exists():
        with open(INFERRED_TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def save_inferred_tasks(tasks: List[Dict]):
    """Save inferred tasks"""
    with open(INFERRED_TASKS_JSON, "w", encoding="utf-8") as fh:
        json.dump(tasks, fh, indent=2)

# ─── Pattern Analysis with LlamaIndex ─────────────────────────────────────────
def analyze_activity_patterns_with_llama(hours: int = 24) -> Dict:
    """Analyze activity patterns using LlamaIndex"""
    activities = load_activities()
    
    if not activities:
        return {}
    
    # Filter recent activities
    cutoff_time = dt.datetime.utcnow() - dt.timedelta(hours=hours)
    recent_activities = [
        act for act in activities
        if dt.datetime.fromisoformat(act["timestamp"]) > cutoff_time
    ]
    
    # Create pattern analysis document
    pattern_text = f"""
Activity Analysis for the last {hours} hours:

Total Activities: {len(recent_activities)}

Activity Breakdown:
"""
    
    category_breakdown = defaultdict(int)
    subcategory_breakdown = defaultdict(int)
    total_duration = 0
    
    for activity in recent_activities:
        # Validate activity has required fields
        if not isinstance(activity, dict):
            print(f"Warning: Skipping invalid activity (not a dict): {activity}")
            continue
        
        if "category" not in activity or "subcategory" not in activity:
            print(f"Warning: Skipping activity missing required fields: {activity}")
            continue
        
        duration = activity.get("duration_seconds", 60)
        total_duration += duration
        
        category = activity["category"]
        subcategory = activity["subcategory"]
        
        category_breakdown[category] += duration
        subcategory_breakdown[subcategory] += duration
        
        pattern_text += f"""
- {category} ({subcategory}): {duration} seconds
  Description: {activity.get('description', 'No description')}
  Text: {activity.get('text_content', 'No text')[:100]}...
"""
    
    pattern_text += f"""
Summary:
- Total time: {total_duration} seconds ({total_duration // 60} minutes)
- Most active category: {max(category_breakdown.items(), key=lambda x: x[1])[0] if category_breakdown else 'None'}
- Most frequent subcategory: {max(subcategory_breakdown.items(), key=lambda x: x[1])[0] if subcategory_breakdown else 'None'}
"""
    
    # Store pattern analysis in vector database
    try:
        index = get_task_index()
        if index:
            pattern_doc = Document(
                text=pattern_text,
                metadata={
                    "analysis_type": "activity_pattern",
                    "time_period_hours": hours,
                    "total_activities": len(recent_activities),
                    "total_duration": total_duration,
                    "timestamp": dt.datetime.utcnow().isoformat()
                }
            )
            index.insert(pattern_doc)
    except Exception as e:
        print(f"Error storing pattern analysis: {e}")
    
    return {
        "total_time": total_duration,
        "category_breakdown": dict(category_breakdown),
        "subcategory_breakdown": dict(subcategory_breakdown),
        "frequent_activities": sorted(subcategory_breakdown.items(), key=lambda x: x[1], reverse=True),
        "pattern_text": pattern_text,
        "recent_activities": recent_activities
    }

# ─── Task Inference with LlamaIndex ───────────────────────────────────────────
def infer_tasks_with_llama(patterns: Dict) -> List[InferredTask]:
    """Use LlamaIndex to infer tasks from activity patterns"""
    try:
        llm, _ = setup_llama_index()
        if not llm:
            return infer_tasks_with_patterns(patterns)
        
        # Create task inference document
        inference_text = f"""
Based on this user activity analysis, suggest 3-5 specific, actionable tasks:

{patterns.get('pattern_text', 'No pattern data available')}

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
        
        # Use LlamaIndex for task inference
        response = llm.complete(inference_text)
        
        # Parse JSON response
        try:
            result = json.loads(response.text.strip())
        except json.JSONDecodeError:
            print(f"JSON parsing error: {response.text}")
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
                created_at=dt.datetime.utcnow().isoformat(),
                metadata={
                    "inference_method": "llama_index",
                    "pattern_analysis": patterns.get("total_time", 0)
                }
            )
            
            inferred_tasks.append(task)
        
        return inferred_tasks
        
    except Exception as e:
        print(f"Error inferring tasks with LlamaIndex: {e}")
        return infer_tasks_with_patterns(patterns)

def infer_tasks_with_patterns(patterns: Dict) -> List[InferredTask]:
    """Fallback task inference using pattern matching"""
    inferred_tasks = []
    existing_inferred = load_inferred_tasks()
    
    # Find frequent activities
    frequent_activities = patterns.get("frequent_activities", [])
    
    for subcategory, duration in frequent_activities[:5]:  # Top 5 activities
        if duration < 1800:  # Less than 30 minutes
            continue
        
        # Find matching patterns
        for category, subcategories in TASK_INFERENCE_PATTERNS.items():
            for pattern_subcategory, task_templates in subcategories.items():
                if pattern_subcategory in subcategory.lower():
                    # Select a task template
                    task_template = task_templates[0]  # Use first template
                    
                    task_id = f"inferred_{len(existing_inferred) + len(inferred_tasks) + 1}"
                    
                    task = InferredTask(
                        task_id=task_id,
                        title=task_template,
                        description=f"Task based on {duration // 60} minutes of {subcategory} activity",
                        category=category,
                        priority="medium" if duration > 3600 else "low",
                        confidence=0.6,
                        reasoning=f"User spent {duration // 60} minutes on {subcategory}",
                        created_at=dt.datetime.utcnow().isoformat(),
                        metadata={
                            "inference_method": "pattern_matching",
                            "activity_duration": duration,
                            "subcategory": subcategory
                        }
                    )
                    
                    inferred_tasks.append(task)
                    break
    
    return inferred_tasks

# ─── Task Storage and Retrieval ───────────────────────────────────────────────
def store_task_in_vector_db(task: InferredTask):
    """Store task in vector database"""
    try:
        index = get_task_index()
        if not index:
            return False
        
        # Create document for vector storage
        doc_text = f"""
Task: {task.title}
Description: {task.description}
Category: {task.category}
Priority: {task.priority}
Reasoning: {task.reasoning}
Confidence: {task.confidence}
"""
        
        doc = Document(
            text=doc_text,
            metadata={
                "task_id": task.task_id,
                "title": task.title,
                "category": task.category,
                "priority": task.priority,
                "confidence": task.confidence,
                "created_at": task.created_at,
                **(task.metadata or {})
            }
        )
        
        # Insert into index
        index.insert(doc)
        
        return True
    except Exception as e:
        print(f"Error storing task in vector DB: {e}")
        return False

def search_similar_tasks(query: str, limit: int = 5) -> List[Dict]:
    """Search for similar tasks using vector similarity"""
    try:
        index = get_task_index()
        if not index:
            return []
        
        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=limit)
        
        # Search
        response = query_engine.query(query)
        
        # Extract results
        results = []
        for node in response.source_nodes:
            results.append({
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, 'score') else 0.0
            })
        
        return results
    except Exception as e:
        print(f"Error searching similar tasks: {e}")
        return []

# ─── Task Management ──────────────────────────────────────────────────────────
def filter_duplicate_tasks(new_tasks: List[InferredTask], existing_tasks: List[Dict]) -> List[InferredTask]:
    """Filter out duplicate tasks"""
    existing_titles = {task.get("title", "").lower() for task in existing_tasks}
    filtered_tasks = []
    
    for task in new_tasks:
        if task.title.lower() not in existing_titles:
            filtered_tasks.append(task)
            existing_titles.add(task.title.lower())
    
    return filtered_tasks

def suggest_productivity_improvements(patterns: Dict) -> List[str]:
    """Suggest productivity improvements based on patterns"""
    suggestions = []
    
    total_time = patterns.get("total_time", 0)
    category_breakdown = patterns.get("category_breakdown", {})
    
    # Time management suggestions
    if total_time > 28800:  # More than 8 hours
        suggestions.append("Consider taking more breaks to maintain productivity")
    
    # Category-specific suggestions
    social_time = category_breakdown.get("social_media", 0)
    if social_time > 7200:  # More than 2 hours
        suggestions.append("Consider limiting social media time to improve focus")
    
    entertainment_time = category_breakdown.get("entertainment", 0)
    if entertainment_time > 14400:  # More than 4 hours
        suggestions.append("Balance entertainment with productive activities")
    
    work_time = category_breakdown.get("work", 0)
    if work_time < 7200:  # Less than 2 hours
        suggestions.append("Consider dedicating more time to work activities")
    
    return suggestions

# ─── Main Analysis Function ───────────────────────────────────────────────────
def analyze_and_infer_tasks_with_llama() -> Tuple[List[InferredTask], List[str]]:
    """Main function to analyze patterns and infer tasks using LlamaIndex"""
    try:
        print(f"🦙 Analyzing activity patterns with LlamaIndex at {dt.datetime.now().isoformat()}")
        
        # Initialize LlamaIndex
        llm, embed_model = setup_llama_index()
        if not llm or not embed_model:
            print("❌ Failed to initialize LlamaIndex")
            return [], []
        
        print("✅ LlamaIndex initialized successfully")
        
        # Analyze patterns
        patterns = analyze_activity_patterns_with_llama(24)  # Last 24 hours
        
        if not patterns:
            print("❌ No activity data found for analysis")
            return [], []
        
        print(f"📊 Found {patterns['total_time'] // 60} minutes of activity")
        
        # Infer tasks using LlamaIndex
        inferred_tasks = infer_tasks_with_llama(patterns)
        
        # Filter duplicates
        existing_tasks = load_existing_tasks()
        filtered_tasks = filter_duplicate_tasks(inferred_tasks, existing_tasks["tasks"])
        
        # Get productivity suggestions
        productivity_suggestions = suggest_productivity_improvements(patterns)
        
        # Store new inferred tasks in vector database
        for task in filtered_tasks:
            store_task_in_vector_db(task)
        
        # Save new inferred tasks
        if filtered_tasks:
            existing_inferred = load_inferred_tasks()
            existing_inferred.extend([asdict(task) for task in filtered_tasks])
            save_inferred_tasks(existing_inferred)
        
        return filtered_tasks, productivity_suggestions
        
    except Exception as e:
        print(f"❌ Error in LlamaIndex task inference: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def main():
    """Main function to run task inference"""
    tasks, suggestions = analyze_and_infer_tasks_with_llama()
    
    if tasks:
        print(f"\n🎯 Generated {len(tasks)} new task suggestions:")
        for i, task in enumerate(tasks, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
            print(f"   {i}. {priority_emoji} {task.title}")
            print(f"      📝 {task.description}")
            print(f"      💭 {task.reasoning}")
            print()
    
    if suggestions:
        print(f"💡 Productivity suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")

if __name__ == "__main__":
    main() 
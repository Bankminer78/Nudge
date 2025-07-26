"""
llama_enhanced_todo_agent.py
───────────────────────────
LlamaIndex-based enhanced todo agent with advanced task management.
Uses vector storage and retrieval for better task organization and user interaction.

Features:
• Vector-based task storage and retrieval
• Advanced task matching using LlamaIndex
• Semantic similarity for task suggestions
• Rich context extraction for task management
• Interactive task management interface
"""

import json
import datetime as dt
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

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

# Import LlamaIndex-based modules
from llama_activity_categorizer import (
    process_activity_with_llama, 
    get_activity_summary,
    search_similar_activities
)
from llama_task_inference_agent import (
    analyze_and_infer_tasks_with_llama,
    load_inferred_tasks,
    search_similar_tasks
)

# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class TodoTask:
    task_id: str
    title: str
    description: str
    category: str
    priority: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TodoDocument(BaseModel):
    """Pydantic model for todo documents"""
    task_id: str
    title: str
    description: str
    category: str
    priority: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = {}

# ─── Config ───────────────────────────────────────────────────────────────────
TASKS_JSON = Path("tasks.json")
VECTOR_DB_DIR = Path("vector_db")
CHROMA_DB_DIR = Path("chroma_db")

# ─── LlamaIndex Setup ─────────────────────────────────────────────────────────
def setup_llama_index():
    """Initialize LlamaIndex with OpenAI components"""
    try:
        # Initialize OpenAI components
        llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
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

def get_todo_vector_store():
    """Get or create vector store for todos"""
    try:
        # Create directories if they don't exist
        VECTOR_DB_DIR.mkdir(exist_ok=True)
        CHROMA_DB_DIR.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        chroma_collection = chroma_client.get_or_create_collection("todos")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        return vector_store
    except Exception as e:
        print(f"Error creating todo vector store: {e}")
        return None

def get_todo_index():
    """Get or create todo index"""
    try:
        vector_store = get_todo_vector_store()
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
        print(f"Error getting todo index: {e}")
        return None

# ─── Task Management ──────────────────────────────────────────────────────────
def load_tasks() -> Dict:
    """Load existing tasks"""
    if TASKS_JSON.exists():
        with open(TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"tasks": []}

def save_tasks(tasks_data: Dict):
    """Save tasks to file"""
    with open(TASKS_JSON, "w", encoding="utf-8") as fh:
        json.dump(tasks_data, fh, indent=2)

def store_todo_in_vector_db(task: TodoTask):
    """Store todo task in vector database"""
    try:
        index = get_todo_index()
        if not index:
            return False
        
        # Create document for vector storage
        doc_text = f"""
Task: {task.title}
Description: {task.description}
Category: {task.category}
Priority: {task.priority}
Status: {task.status}
Created: {task.created_at}
"""
        
        if task.completed_at:
            doc_text += f"Completed: {task.completed_at}\n"
        
        doc = Document(
            text=doc_text,
            metadata={
                "task_id": task.task_id,
                "title": task.title,
                "category": task.category,
                "priority": task.priority,
                "status": task.status,
                "created_at": task.created_at,
                "completed_at": task.completed_at or "",
                **(task.metadata or {})
            }
        )
        
        # Insert into index
        index.insert(doc)
        
        return True
    except Exception as e:
        print(f"Error storing todo in vector DB: {e}")
        return False

def search_todos_with_llama(query: str, limit: int = 5) -> List[Dict]:
    """Search for todos using LlamaIndex vector similarity"""
    try:
        index = get_todo_index()
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
        print(f"Error searching todos: {e}")
        return []

# ─── Task Matching with LlamaIndex ────────────────────────────────────────────
def match_activity_to_tasks_with_llama(activity_description: str, existing_tasks: List[Dict]) -> List[Tuple[Dict, float]]:
    """Match activity to existing tasks using LlamaIndex"""
    try:
        llm, _ = setup_llama_index()
        if not llm:
            return []
        
        # Create matching document
        matching_text = f"""
Activity: {activity_description}

Existing Tasks:
"""
        
        for i, task in enumerate(existing_tasks, 1):
            matching_text += f"""
{i}. {task.get('title', 'No title')}
   Description: {task.get('description', 'No description')}
   Category: {task.get('category', 'No category')}
   Status: {task.get('status', 'pending')}
"""
        
        matching_text += f"""
Based on the activity description, which existing tasks are most relevant?
Consider:
1. Semantic similarity between activity and task
2. Category matching
3. Context relevance
4. Task status (prefer pending tasks)

Respond with JSON format:
{{
    "matches": [
        {{
            "task_index": 0,
            "relevance_score": 0.95,
            "reasoning": "Why this task matches the activity"
        }}
    ]
}}
"""
        
        # Use LlamaIndex for matching
        response = llm.complete(matching_text)
        
        # Parse JSON response
        try:
            result = json.loads(response.text.strip())
        except json.JSONDecodeError:
            print(f"JSON parsing error: {response.text}")
            return []
        
        # Validate and return matches
        matches = []
        if "matches" in result:
            for match in result["matches"]:
                if "task_index" in match and "relevance_score" in match:
                    task_index = match["task_index"]
                    if 0 <= task_index < len(existing_tasks):
                        matches.append((existing_tasks[task_index], match["relevance_score"]))
        
        return matches
        
    except Exception as e:
        print(f"Error matching activity to tasks: {e}")
        return []

def create_task_from_activity_with_llama(activity_description: str) -> Optional[TodoTask]:
    """Create a new task from activity using LlamaIndex"""
    try:
        llm, _ = setup_llama_index()
        if not llm:
            return None
        
        # Create task generation document
        task_text = f"""
Based on this activity, create a new todo task:

Activity: {activity_description}

Create a task that:
1. Is specific and actionable
2. Has a clear title and description
3. Is categorized appropriately
4. Has appropriate priority
5. Captures the intent of the activity

Respond with JSON format:
{{
    "title": "Specific task title",
    "description": "Detailed task description with actionable steps",
    "category": "work|personal|learning|shopping|health|other",
    "priority": "high|medium|low",
    "reasoning": "Why this task should be created"
}}
"""
        
        # Use LlamaIndex for task creation
        response = llm.complete(task_text)
        
        # Parse JSON response
        try:
            result = json.loads(response.text.strip())
        except json.JSONDecodeError:
            print(f"JSON parsing error: {response.text}")
            return None
        
        # Validate required fields
        required_fields = ["title", "description", "category", "priority"]
        for field in required_fields:
            if field not in result:
                print(f"Missing field '{field}' in task creation response")
                return None
        
        # Create task
        tasks_data = load_tasks()
        task_id = f"task_{len(tasks_data['tasks']) + 1}"
        
        task = TodoTask(
            task_id=task_id,
            title=result["title"],
            description=result["description"],
            category=result["category"],
            priority=result["priority"],
            status="pending",
            created_at=dt.datetime.utcnow().isoformat(),
            metadata={
                "created_from_activity": True,
                "activity_description": activity_description,
                "reasoning": result.get("reasoning", ""),
                "creation_method": "llama_index"
            }
        )
        
        # Store in vector database
        store_todo_in_vector_db(task)
        
        return task
        
    except Exception as e:
        print(f"Error creating task from activity: {e}")
        return None

# ─── Interactive Interface ────────────────────────────────────────────────────
def display_tasks(tasks: List[Dict], title: str = "Tasks"):
    """Display tasks in a formatted way"""
    print(f"\n📋 {title}")
    print("=" * 60)
    
    if not tasks:
        print("   No tasks found")
        return
    
    for i, task in enumerate(tasks, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(task.get("status", "pending"), "❓")
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.get("priority", "medium"), "⚪")
        
        print(f"{i:2d}. {status_emoji} {priority_emoji} {task.get('title', 'No title')}")
        print(f"     📝 {task.get('description', 'No description')[:80]}...")
        print(f"     🏷️  {task.get('category', 'No category')} | {task.get('status', 'pending')}")
        print()

def display_inferred_tasks(tasks: List[Dict]):
    """Display inferred tasks"""
    print(f"\n🎯 Inferred Task Suggestions")
    print("=" * 60)
    
    if not tasks:
        print("   No inferred tasks available")
        return
    
    for i, task in enumerate(tasks, 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.get("priority", "medium"), "⚪")
        
        print(f"{i:2d}. {priority_emoji} {task.get('title', 'No title')}")
        print(f"     📝 {task.get('description', 'No description')[:80]}...")
        print(f"     💭 {task.get('reasoning', 'No reasoning')[:80]}...")
        print(f"     🎯 Confidence: {task.get('confidence', 0):.2f}")
        print()

def interactive_task_management():
    """Interactive task management interface"""
    print("🦙 LlamaIndex Enhanced Todo Agent")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. View all tasks")
        print("2. View pending tasks")
        print("3. View inferred task suggestions")
        print("4. Process latest activity")
        print("5. Search tasks")
        print("6. Add inferred task to todos")
        print("7. Mark task as complete")
        print("8. Run task inference")
        print("9. Show activity summary")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-9): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            tasks_data = load_tasks()
            display_tasks(tasks_data["tasks"], "All Tasks")
        elif choice == "2":
            tasks_data = load_tasks()
            pending_tasks = [t for t in tasks_data["tasks"] if t.get("status") == "pending"]
            display_tasks(pending_tasks, "Pending Tasks")
        elif choice == "3":
            inferred_tasks = load_inferred_tasks()
            pending_inferred = [t for t in inferred_tasks if not t.get("added_to_todos", False)]
            display_inferred_tasks(pending_inferred)
        elif choice == "4":
            process_latest_activity()
        elif choice == "5":
            search_query = input("Enter search query: ").strip()
            if search_query:
                search_results = search_todos_with_llama(search_query)
                if search_results:
                    print(f"\n🔍 Search Results for '{search_query}':")
                    for i, result in enumerate(search_results, 1):
                        print(f"{i}. {result['metadata'].get('title', 'No title')}")
                        print(f"   Score: {result['score']:.3f}")
                        print()
                else:
                    print("❌ No search results found")
        elif choice == "6":
            add_inferred_task_to_todos()
        elif choice == "7":
            mark_task_complete()
        elif choice == "8":
            run_task_inference()
        elif choice == "9":
            show_activity_summary()
        else:
            print("❌ Invalid choice")

def process_latest_activity():
    """Process latest activity and suggest tasks"""
    print("\n🔄 Processing latest activity...")
    
    activity = process_activity_with_llama()
    
    if not activity:
        print("❌ No new activity found")
        return
    
    print(f"✅ Activity: {activity.category} - {activity.subcategory}")
    print(f"   Description: {activity.description}")
    
    # Load existing tasks
    tasks_data = load_tasks()
    existing_tasks = tasks_data["tasks"]
    
    # Match activity to existing tasks
    matches = match_activity_to_tasks_with_llama(activity.description, existing_tasks)
    
    if matches:
        print(f"\n🎯 Found {len(matches)} matching tasks:")
        for task, score in matches[:3]:  # Show top 3
            print(f"   • {task.get('title', 'No title')} (relevance: {score:.2f})")
    
    # Ask if user wants to create a new task
    create_new = input("\nCreate new task from this activity? (y/n): ").strip().lower()
    
    if create_new == 'y':
        new_task = create_task_from_activity_with_llama(activity.description)
        if new_task:
            # Add to tasks
            tasks_data["tasks"].append(asdict(new_task))
            save_tasks(tasks_data)
            print(f"✅ Created new task: {new_task.title}")
        else:
            print("❌ Failed to create task")

def add_inferred_task_to_todos():
    """Add an inferred task to the todo list"""
    inferred_tasks = load_inferred_tasks()
    pending_inferred = [t for t in inferred_tasks if not t.get("added_to_todos", False)]
    
    if not pending_inferred:
        print("❌ No pending inferred tasks available")
        return
    
    display_inferred_tasks(pending_inferred)
    
    try:
        choice = int(input("\nEnter task number to add (0 to cancel): ").strip())
        if choice == 0:
            return
        if 1 <= choice <= len(pending_inferred):
            selected_task = pending_inferred[choice - 1]
            
            # Create todo task
            tasks_data = load_tasks()
            task_id = f"task_{len(tasks_data['tasks']) + 1}"
            
            todo_task = TodoTask(
                task_id=task_id,
                title=selected_task["title"],
                description=selected_task["description"],
                category=selected_task.get("category", "inferred"),
                priority=selected_task["priority"],
                status="pending",
                created_at=dt.datetime.utcnow().isoformat(),
                metadata={
                    "from_inferred": True,
                    "original_task_id": selected_task["task_id"],
                    "reasoning": selected_task["reasoning"],
                    "confidence": selected_task["confidence"]
                }
            )
            
            # Add to tasks
            tasks_data["tasks"].append(asdict(todo_task))
            save_tasks(tasks_data)
            
            # Store in vector database
            store_todo_in_vector_db(todo_task)
            
            # Mark as added
            selected_task["added_to_todos"] = True
            save_inferred_tasks(inferred_tasks)
            
            print(f"✅ Added task: {todo_task.title}")
        else:
            print("❌ Invalid task number")
    except ValueError:
        print("❌ Invalid input")

def mark_task_complete():
    """Mark a task as complete"""
    tasks_data = load_tasks()
    pending_tasks = [t for t in tasks_data["tasks"] if t.get("status") == "pending"]
    
    if not pending_tasks:
        print("❌ No pending tasks available")
        return
    
    display_tasks(pending_tasks, "Pending Tasks")
    
    try:
        choice = int(input("\nEnter task number to mark complete (0 to cancel): ").strip())
        if choice == 0:
            return
        if 1 <= choice <= len(pending_tasks):
            selected_task = pending_tasks[choice - 1]
            
            # Find and update task
            for task in tasks_data["tasks"]:
                if task["task_id"] == selected_task["task_id"]:
                    task["status"] = "completed"
                    task["completed_at"] = dt.datetime.utcnow().isoformat()
                    break
            
            save_tasks(tasks_data)
            print(f"✅ Marked task as complete: {selected_task['title']}")
        else:
            print("❌ Invalid task number")
    except ValueError:
        print("❌ Invalid input")

def run_task_inference():
    """Run task inference analysis"""
    print("\n🦙 Running task inference with LlamaIndex...")
    
    tasks, suggestions = analyze_and_infer_tasks_with_llama()
    
    if tasks:
        print(f"\n🎯 Generated {len(tasks)} new task suggestions:")
        for i, task in enumerate(tasks, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
            print(f"   {i}. {priority_emoji} {task.title}")
            print(f"      📝 {task.description[:80]}...")
    else:
        print("   ⏳ No new task suggestions")
    
    if suggestions:
        print(f"\n💡 Productivity suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")

def show_activity_summary():
    """Show activity summary"""
    print("\n📊 Activity Summary")
    print("=" * 30)
    
    summary = get_activity_summary(1)  # Last hour
    print(f"Last hour: {summary['total_activities']} activities")
    
    for category, data in summary["categories"].items():
        duration_minutes = data["total_duration"] // 60
        print(f"   {category}: {data['count']} activities ({duration_minutes} min)")

# ─── Continuous Monitoring ────────────────────────────────────────────────────
def continuous_monitoring_with_llama():
    """Continuous monitoring mode using LlamaIndex"""
    print("🦙 Starting LlamaIndex Continuous Monitoring...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    last_activity_time = dt.datetime.now()
    last_inference_time = dt.datetime.now()
    
    # Configuration (in milliseconds)
    ACTIVITY_INTERVAL = 5000  # Process activity every 5 seconds
    INFERENCE_INTERVAL = 15000  # Run inference every 15 seconds
    
    try:
        while True:
            current_time = dt.datetime.now()
            
            # Process new activity
            if (current_time - last_activity_time).total_seconds() * 1000 >= ACTIVITY_INTERVAL:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Processing activities...")
                
                try:
                    activity = process_activity_with_llama()
                    if activity:
                        print(f"✅ Categorized: {activity.category} - {activity.subcategory}")
                        if activity.description:
                            print(f"   📝 {activity.description[:80]}...")
                    else:
                        print("   ⏳ No new activity found")
                except Exception as e:
                    print(f"   ❌ Error processing activity: {e}")
                
                last_activity_time = current_time
            
            # Analyze patterns and suggest tasks
            if (current_time - last_inference_time).total_seconds() * 1000 >= INFERENCE_INTERVAL:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Analyzing patterns...")
                
                try:
                    tasks, suggestions = analyze_and_infer_tasks_with_llama()
                    
                    if tasks:
                        print(f"🎯 Generated {len(tasks)} new task suggestions:")
                        for i, task in enumerate(tasks[:3], 1):  # Show top 3
                            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
                            print(f"   {i}. {priority_emoji} {task.title}")
                        if len(tasks) > 3:
                            print(f"   ... and {len(tasks) - 3} more")
                    
                    if suggestions:
                        print(f"💡 Productivity suggestions:")
                        for suggestion in suggestions:
                            print(f"   • {suggestion}")
                except Exception as e:
                    print(f"   ❌ Error in task inference: {e}")
                
                last_inference_time = current_time
                print("=" * 60)
            
            time.sleep(1)  # Check every 1 second
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        print("📊 Final Summary:")
        try:
            summary = get_activity_summary(1)
            print(f"   Total activities in last hour: {summary['total_activities']}")
        except:
            pass
        print("👋 Goodbye!")

def main():
    """Main function"""
    print("🦙 LlamaIndex Enhanced Todo Agent")
    print("=" * 50)
    
    # Initialize LlamaIndex
    llm, embed_model = setup_llama_index()
    if not llm or not embed_model:
        print("❌ Failed to initialize LlamaIndex")
        return
    
    print("✅ LlamaIndex initialized successfully")
    
    # Show options
    print("\nChoose mode:")
    print("1. Interactive mode")
    print("2. Continuous monitoring")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        interactive_task_management()
    elif choice == "2":
        continuous_monitoring_with_llama()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 
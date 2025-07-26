"""
enhanced_todo_agent.py
─────────────────────
Enhanced todo agent that combines activity categorization, task inference, and traditional todo management.

Features:
• Processes screen captures and categorizes activities
• Infers potential tasks from behavior patterns
• Manages existing tasks and suggests new ones
• Provides productivity insights and recommendations

Usage:
    export OPENAI_API_KEY=sk-...
    python enhanced_todo_agent.py
"""

import json
import os
import glob
import datetime as dt
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import openai

import dotenv
dotenv.load_dotenv()

# Import our new modules
from activity_categorizer import process_activity, get_activity_summary
from task_inference_agent import analyze_and_infer_tasks, load_inferred_tasks

# ─── Config ───────────────────────────────────────────────────────────────────
CAP_DIR = Path("captures")
TASKS_JSON = Path("tasks.json")
INFERRED_TASKS_JSON = Path("inferred_tasks.json")
ACTIVITIES_JSON = Path("user_activities.json")

# ─── Enhanced Task Management ─────────────────────────────────────────────────
def load_tasks():
    """Load existing tasks"""
    if TASKS_JSON.exists():
        with open(TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"tasks": []}

def save_tasks(data):
    """Save tasks to file"""
    with open(TASKS_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

def add_inferred_task_to_todos(task_id: str) -> bool:
    """Add an inferred task to the main todo list"""
    try:
        # Load inferred tasks
        inferred_tasks = load_inferred_tasks()
        target_task = None
        
        for task in inferred_tasks:
            if task["task_id"] == task_id:
                target_task = task
                break
        
        if not target_task:
            print(f"❌ Inferred task {task_id} not found")
            return False
        
        # Load existing todos
        todos_data = load_tasks()
        
        # Create new todo from inferred task
        new_todo = {
            "id": f"todo_{len(todos_data['tasks']) + 1}",
            "intent": target_task["title"],
            "description": target_task["description"],
            "category": target_task["category"],
            "priority": target_task["priority"],
            "source": "inferred",
            "inferred_task_id": task_id,
            "created_at": dt.datetime.utcnow().isoformat(),
            "last_seen": dt.datetime.utcnow().isoformat()
        }
        
        todos_data["tasks"].append(new_todo)
        save_tasks(todos_data)
        
        print(f"✅ Added inferred task to todos: {target_task['title']}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding inferred task: {e}")
        return False

def display_inferred_tasks():
    """Display all inferred tasks with option to add to todos"""
    inferred_tasks = load_inferred_tasks()
    
    if not inferred_tasks:
        print("📝 No inferred tasks available")
        return
    
    print(f"\n🎯 Available Inferred Tasks ({len(inferred_tasks)}):")
    
    for i, task in enumerate(inferred_tasks, 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task["priority"], "⚪")
        status_emoji = "✅" if task.get("added_to_todos") else "⏳"
        
        print(f"   {i}. {priority_emoji} {task['title']} {status_emoji}")
        print(f"      📝 {task['description']}")
        print(f"      💭 {task['reasoning']}")
        print(f"      📅 Created: {task['created_at'][:19]}")
        print()

def interactive_task_management():
    """Interactive interface for managing inferred tasks"""
    while True:
        print("\n" + "="*60)
        print("🎯 ENHANCED TODO MANAGEMENT")
        print("="*60)
        
        # Show current todos
        todos_data = load_tasks()
        print(f"📋 Current Todos: {len(todos_data['tasks'])}")
        
        # Show inferred tasks
        inferred_tasks = load_inferred_tasks()
        print(f"🎯 Inferred Tasks: {len(inferred_tasks)}")
        
        # Show recent activity summary
        activity_summary = get_activity_summary(1)  # Last hour
        print(f"📊 Recent Activity: {activity_summary['total_activities']} activities")
        
        print("\nOptions:")
        print("1. View all inferred tasks")
        print("2. Add inferred task to todos")
        print("3. Analyze patterns and generate new suggestions")
        print("4. View activity summary")
        print("5. View current todos")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            display_inferred_tasks()
            
        elif choice == "2":
            display_inferred_tasks()
            if inferred_tasks:
                try:
                    task_num = int(input("Enter task number to add to todos: ")) - 1
                    if 0 <= task_num < len(inferred_tasks):
                        task_id = inferred_tasks[task_num]["task_id"]
                        if add_inferred_task_to_todos(task_id):
                            # Mark as added
                            inferred_tasks[task_num]["added_to_todos"] = True
                            with open(INFERRED_TASKS_JSON, "w", encoding="utf-8") as fh:
                                json.dump(inferred_tasks, fh, indent=2)
                    else:
                        print("❌ Invalid task number")
                except ValueError:
                    print("❌ Please enter a valid number")
                    
        elif choice == "3":
            print("🔍 Analyzing patterns and generating suggestions...")
            tasks, suggestions = analyze_and_infer_tasks()
            if tasks:
                print(f"✅ Generated {len(tasks)} new task suggestions")
            if suggestions:
                print(f"💡 {len(suggestions)} productivity suggestions")
                
        elif choice == "4":
            hours = input("Enter hours to analyze (default 24): ").strip()
            try:
                hours = int(hours) if hours else 24
                summary = get_activity_summary(hours)
                print(f"\n📊 Activity Summary (Last {hours} hours):")
                print(f"Total activities: {summary['total_activities']}")
                
                for category, data in summary["categories"].items():
                    duration_minutes = data["total_duration"] // 60
                    print(f"   {category}: {data['count']} activities ({duration_minutes} min)")
                    
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif choice == "5":
            todos_data = load_tasks()
            if todos_data["tasks"]:
                print(f"\n📋 Current Todos ({len(todos_data['tasks'])}):")
                for i, task in enumerate(todos_data["tasks"], 1):
                    print(f"   {i}. {task['intent']}")
                    if task.get("description"):
                        print(f"      📝 {task['description']}")
                    print(f"      📅 Created: {task['created_at'][:19]}")
                    print()
            else:
                print("📋 No todos found")
                
        elif choice == "6":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def continuous_monitoring():
    """Continuous monitoring mode - processes activities and suggests tasks"""
    print("🔄 Starting continuous monitoring mode...")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
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
                    # Process activity
                    activity = process_activity()
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
                    tasks, suggestions = analyze_and_infer_tasks()
                    
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
                print("=" * 50)
            
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
    """Main function with enhanced todo management"""
    print("🚀 Enhanced Todo Agent")
    print("=" * 40)
    
    # Check if we have activity data
    if not ACTIVITIES_JSON.exists():
        print("📝 No activity data found. Starting activity categorization...")
        activity = process_activity()
        if activity:
            print(f"✅ Initial activity processed: {activity.category}")
        else:
            print("❌ No initial activity found")
    
    # Check if we have inferred tasks
    if not INFERRED_TASKS_JSON.exists():
        print("🔍 No inferred tasks found. Running initial analysis...")
        tasks, suggestions = analyze_and_infer_tasks()
        if tasks:
            print(f"✅ Generated {len(tasks)} initial task suggestions")
    
    print("\nChoose mode:")
    print("1. Interactive Management")
    print("2. Continuous Monitoring")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        interactive_task_management()
    elif choice == "2":
        continuous_monitoring()
    else:
        print("❌ Invalid choice. Starting interactive mode...")
        interactive_task_management()

if __name__ == "__main__":
    main()
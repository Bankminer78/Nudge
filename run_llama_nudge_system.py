"""
run_llama_nudge_system.py
─────────────────────────
Main runner script for the LlamaIndex-based Nudge system.
Orchestrates all LlamaIndex components and provides different operation modes.

Features:
• LlamaIndex-based activity categorization
• Advanced task inference with vector storage
• Enhanced todo management with semantic search
• Continuous monitoring with real-time processing
• Multiple operation modes for different use cases
"""

import sys
import datetime as dt
import time
from pathlib import Path
import dotenv
dotenv.load_dotenv()

# Import LlamaIndex-based modules
from llama_activity_categorizer import (
    process_activity_with_llama,
    get_activity_summary,
    setup_llama_index as setup_activity_llama
)
from llama_task_inference_agent import (
    analyze_and_infer_tasks_with_llama,
    load_inferred_tasks,
    setup_llama_index as setup_task_llama
)
from llama_enhanced_todo_agent import (
    interactive_task_management,
    continuous_monitoring_with_llama,
    setup_llama_index as setup_todo_llama
)

# ─── System Status ────────────────────────────────────────────────────────────
def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check for required packages
    try:
        import llama_index
        print("✅ LlamaIndex")
    except ImportError:
        print("❌ LlamaIndex not found. Install with: pip install llama-index")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB")
    except ImportError:
        print("❌ ChromaDB not found. Install with: pip install chromadb")
        return False
    
    try:
        import openai
        print("✅ OpenAI")
    except ImportError:
        print("❌ OpenAI not found. Install with: pip install openai")
        return False
    
    # Check for environment variables
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    else:
        print("✅ OpenAI API Key")
    
    # Check for required directories
    captures_dir = Path("captures")
    if not captures_dir.exists():
        print("⚠️  Captures directory not found. Creating...")
        captures_dir.mkdir(exist_ok=True)
    
    print("✅ All dependencies satisfied")
    return True

def show_system_status():
    """Show current system status"""
    print("\n📊 System Status:")
    print("=" * 40)
    
    # Check data files
    files_to_check = [
        ("user_activities.json", "Activity Data"),
        ("tasks.json", "Task Data"),
        ("inferred_tasks.json", "Inferred Tasks"),
        ("vector_db", "Vector Database"),
        ("chroma_db", "ChromaDB")
    ]
    
    for file_path, description in files_to_check:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"✅ {description}: {size} bytes")
            else:
                # Directory
                count = len(list(path.rglob("*")))
                print(f"✅ {description}: {count} items")
        else:
            print(f"❌ {description}: Not found")
    
    # Check captures
    captures_dir = Path("captures")
    if captures_dir.exists():
        capture_files = list(captures_dir.glob("*"))
        txt_files = [f for f in capture_files if f.suffix == ".txt"]
        img_files = [f for f in capture_files if f.suffix in [".png", ".jpg", ".jpeg"]]
        print(f"✅ Captures: {len(txt_files)} text, {len(img_files)} images")
    else:
        print("❌ Captures: Directory not found")

# ─── Operation Modes ──────────────────────────────────────────────────────────
def run_activity_categorization():
    """Run activity categorization only"""
    print("\n🦙 Running LlamaIndex Activity Categorization...")
    print("=" * 60)
    
    # Initialize LlamaIndex
    llm, embed_model = setup_activity_llama()
    if not llm or not embed_model:
        print("❌ Failed to initialize LlamaIndex")
        return
    
    print("✅ LlamaIndex initialized successfully")
    
    # Process activity
    activity = process_activity_with_llama()
    
    if activity:
        print(f"✅ Categorized activity: {activity.category} - {activity.subcategory}")
        print(f"   Description: {activity.description}")
        print(f"   Confidence: {activity.confidence:.2f}")
        print(f"   Processing: LlamaIndex")
        
        # Show recent summary
        summary = get_activity_summary(1)
        print(f"\n📊 Last hour: {summary['total_activities']} activities")
        
        for category, data in summary["categories"].items():
            duration_minutes = data["total_duration"] // 60
            print(f"   {category}: {data['count']} activities ({duration_minutes} min)")
    else:
        print("❌ No new activity found")

def run_task_inference():
    """Run task inference only"""
    print("\n🦙 Running LlamaIndex Task Inference...")
    print("=" * 60)
    
    # Initialize LlamaIndex
    llm, embed_model = setup_task_llama()
    if not llm or not embed_model:
        print("❌ Failed to initialize LlamaIndex")
        return
    
    print("✅ LlamaIndex initialized successfully")
    
    # Run inference
    tasks, suggestions = analyze_and_infer_tasks_with_llama()
    
    if tasks:
        print(f"\n🎯 Generated {len(tasks)} new task suggestions:")
        for i, task in enumerate(tasks, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
            print(f"   {i}. {priority_emoji} {task.title}")
            print(f"      📝 {task.description}")
            print(f"      💭 {task.reasoning}")
            print()
    else:
        print("   ⏳ No new task suggestions")
    
    if suggestions:
        print(f"💡 Productivity suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")

def run_enhanced_todo():
    """Run enhanced todo agent"""
    print("\n🦙 Running LlamaIndex Enhanced Todo Agent...")
    print("=" * 60)
    
    # Initialize LlamaIndex
    llm, embed_model = setup_todo_llama()
    if not llm or not embed_model:
        print("❌ Failed to initialize LlamaIndex")
        return
    
    print("✅ LlamaIndex initialized successfully")
    
    # Run interactive interface
    interactive_task_management()

def run_continuous_monitoring():
    """Run continuous monitoring with LlamaIndex"""
    print("\n🦙 Starting LlamaIndex Continuous Monitoring...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Initialize LlamaIndex for all components
    activity_llm, activity_embed = setup_activity_llama()
    task_llm, task_embed = setup_task_llama()
    todo_llm, todo_embed = setup_todo_llama()
    
    if not all([activity_llm, task_llm, todo_llm]):
        print("❌ Failed to initialize LlamaIndex components")
        return
    
    print("✅ All LlamaIndex components initialized")
    
    last_activity_time = dt.datetime.now()
    last_inference_time = dt.datetime.now()
    last_summary_time = dt.datetime.now()
    
    # Configuration (in milliseconds)
    ACTIVITY_INTERVAL = 2000  # Process activity every 2 seconds
    INFERENCE_INTERVAL = 2000  # Run inference every 2 seconds
    SUMMARY_INTERVAL = 30000  # Show summary every 30 seconds
    
    print(f"📊 Monitoring Configuration:")
    print(f"   Activity processing: Every {ACTIVITY_INTERVAL/1000:.1f} seconds")
    print(f"   Task inference: Every {INFERENCE_INTERVAL/1000:.1f} seconds")
    print(f"   Summary display: Every {SUMMARY_INTERVAL/1000:.1f} seconds")
    print("=" * 60)
    
    try:
        while True:
            current_time = dt.datetime.now()
            
            # Process activity
            if (current_time - last_activity_time).total_seconds() * 1000 >= ACTIVITY_INTERVAL:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Processing activity...")
                try:
                    activity = process_activity_with_llama()
                    if activity:
                        print(f"✅ Categorized: {activity.category} - {activity.subcategory}")
                        if activity.description:
                            print(f"   📝 {activity.description[:100]}...")
                    else:
                        print("   ⏳ No new activity found")
                except Exception as e:
                    print(f"   ❌ Error processing activity: {e}")
                last_activity_time = current_time
            
            # Run task inference
            if (current_time - last_inference_time).total_seconds() * 1000 >= INFERENCE_INTERVAL:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Running task inference...")
                try:
                    tasks, suggestions = analyze_and_infer_tasks_with_llama()
                    if tasks:
                        print(f"🎯 Generated {len(tasks)} new task suggestions:")
                        for i, task in enumerate(tasks[:3], 1):  # Show top 3
                            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
                            print(f"   {i}. {priority_emoji} {task.title}")
                        if len(tasks) > 3:
                            print(f"   ... and {len(tasks) - 3} more")
                    else:
                        print("   ⏳ No new task suggestions")
                    
                    if suggestions:
                        print(f"💡 Productivity suggestions:")
                        for suggestion in suggestions:
                            print(f"   • {suggestion}")
                except Exception as e:
                    print(f"   ❌ Error in task inference: {e}")
                last_inference_time = current_time
            
            # Show activity summary
            if (current_time - last_summary_time).total_seconds() * 1000 >= SUMMARY_INTERVAL:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Activity Summary:")
                try:
                    summary = get_activity_summary(1)  # Last hour
                    print(f"📊 Last hour: {summary['total_activities']} activities")
                    
                    for category, data in summary["categories"].items():
                        duration_minutes = data["total_duration"] // 60
                        print(f"   {category}: {data['count']} activities ({duration_minutes} min)")
                    
                    # Show pending inferred tasks
                    inferred_tasks = load_inferred_tasks()
                    pending_tasks = [t for t in inferred_tasks if not t.get("added_to_todos", False)]
                    if pending_tasks:
                        print(f"🎯 Pending task suggestions: {len(pending_tasks)}")
                except Exception as e:
                    print(f"   ❌ Error generating summary: {e}")
                last_summary_time = current_time
                
                print("=" * 60)
            
            # Sleep for a shorter interval for more responsive monitoring
            time.sleep(1)  # Check every 1 second
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        print("📊 Final Summary:")
        try:
            summary = get_activity_summary(1)
            print(f"   Total activities in last hour: {summary['total_activities']}")
            inferred_tasks = load_inferred_tasks()
            pending_tasks = [t for t in inferred_tasks if not t.get("added_to_todos", False)]
            print(f"   Pending task suggestions: {len(pending_tasks)}")
        except:
            pass
        print("👋 Goodbye!")

def run_background_monitoring():
    """Run background monitoring with minimal output"""
    print("\n🦙 Starting LlamaIndex Background Monitoring...")
    print("Running silently in background. Check logs for activity.")
    print("Press Ctrl+C to stop")
    
    # Initialize LlamaIndex for all components
    activity_llm, activity_embed = setup_activity_llama()
    task_llm, task_embed = setup_task_llama()
    
    if not all([activity_llm, task_llm]):
        print("❌ Failed to initialize LlamaIndex components")
        return
    
    print("✅ LlamaIndex components initialized")
    
    last_activity_time = dt.datetime.now()
    last_inference_time = dt.datetime.now()
    
    # Configuration for background mode (in milliseconds)
    ACTIVITY_INTERVAL = 10000  # Process activity every 10 seconds
    INFERENCE_INTERVAL = 30000  # Run inference every 30 seconds
    
    try:
        while True:
            current_time = dt.datetime.now()
            
            # Process activity silently
            if (current_time - last_activity_time).total_seconds() * 1000 >= ACTIVITY_INTERVAL:
                try:
                    activity = process_activity_with_llama()
                    if activity:
                        print(f"[{current_time.strftime('%H:%M:%S')}] Activity: {activity.category} - {activity.subcategory}")
                except Exception as e:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Activity error: {e}")
                last_activity_time = current_time
            
            # Run task inference silently
            if (current_time - last_inference_time).total_seconds() * 1000 >= INFERENCE_INTERVAL:
                try:
                    tasks, suggestions = analyze_and_infer_tasks_with_llama()
                    if tasks:
                        print(f"[{current_time.strftime('%H:%M:%S')}] Generated {len(tasks)} task suggestions")
                    if suggestions:
                        print(f"[{current_time.strftime('%H:%M:%S')}] {len(suggestions)} productivity suggestions")
                except Exception as e:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Inference error: {e}")
                last_inference_time = current_time
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Background monitoring stopped")

def run_full_system():
    """Run the full LlamaIndex-based Nudge system"""
    print("\n🦙 Running Full LlamaIndex Nudge System...")
    print("=" * 60)
    
    # Initialize all LlamaIndex components
    print("🔄 Initializing LlamaIndex components...")
    
    activity_llm, activity_embed = setup_activity_llama()
    task_llm, task_embed = setup_task_llama()
    todo_llm, todo_embed = setup_todo_llama()
    
    if not all([activity_llm, task_llm, todo_llm]):
        print("❌ Failed to initialize LlamaIndex components")
        return
    
    print("✅ All LlamaIndex components initialized successfully")
    
    # Process initial activity
    print("\n🔄 Processing initial activity...")
    activity = process_activity_with_llama()
    
    if activity:
        print(f"✅ Initial activity: {activity.category} - {activity.subcategory}")
    
    # Run initial task inference
    print("\n🔄 Running initial task inference...")
    tasks, suggestions = analyze_and_infer_tasks_with_llama()
    
    if tasks:
        print(f"✅ Generated {len(tasks)} initial task suggestions")
    
    # Show system summary
    print("\n📊 System Summary:")
    summary = get_activity_summary(24)  # Last 24 hours
    print(f"   Total activities: {summary['total_activities']}")
    
    for category, data in summary["categories"].items():
        duration_minutes = data["total_duration"] // 60
        print(f"   {category}: {data['count']} activities ({duration_minutes} min)")
    
    print("\n🎯 System ready! Use interactive mode for task management.")

# ─── Main Function ────────────────────────────────────────────────────────────
def main():
    """Main function"""
    print("🦙 LLAMAINDEX NUDGE SYSTEM")
    print("=" * 50)
    print("Advanced activity tracking and task inference with LlamaIndex")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please fix the above issues before running the system.")
        return
    
    # Show system status
    show_system_status()
    
    # Get mode from command line or user input
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not mode:
        print("\nChoose mode:")
        print("1. categorize - Activity categorization only")
        print("2. infer - Task inference only")
        print("3. enhanced - Enhanced todo agent")
        print("4. monitor - Continuous monitoring loop")
        print("5. background - Background monitoring (minimal output)")
        print("6. all - Full system (default)")
        
        choice = input("\nEnter choice (1-6, default 6): ").strip()
        
        mode_map = {
            "1": "categorize",
            "2": "infer",
            "3": "enhanced",
            "4": "monitor",
            "5": "background",
            "6": "all"
        }
        mode = mode_map.get(choice, "all")
    
    # Run selected mode
    if mode == "categorize":
        run_activity_categorization()
    elif mode == "infer":
        run_task_inference()
    elif mode == "enhanced":
        run_enhanced_todo()
    elif mode == "monitor":
        run_continuous_monitoring()
    elif mode == "background":
        run_background_monitoring()
    elif mode == "all":
        run_full_system()
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: categorize, infer, enhanced, monitor, background, all")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
run_nudge_system.py
──────────────────
Main runner script for the Nudge system.
Orchestrates activity categorization, task inference, and todo management.

Usage:
    export OPENAI_API_KEY=sk-...
    python run_nudge_system.py [mode]

Modes:
    - categorize: Run activity categorization only
    - infer: Run task inference only  
    - enhanced: Run enhanced todo agent
    - monitor: Run continuous monitoring
    - all: Run full system (default)
"""

import sys
import time
import datetime as dt
from pathlib import Path
import dotenv
dotenv.load_dotenv()

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import openai
        print("✅ OpenAI library found")
    except ImportError:
        print("❌ OpenAI library not found. Install with: pip install openai")
        return False
    
    try:
        import requests
        print("✅ Requests library found")
    except ImportError:
        print("❌ Requests library not found. Install with: pip install requests")
        return False
    
    # Check if OpenAI API key is set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        return False
    
    print("✅ OpenAI API key found")
    return True

def run_activity_categorization():
    """Run activity categorization"""
    print("\n🔍 Running Activity Categorization...")
    try:
        from activity_categorizer import process_activity, get_activity_summary
        
        activity = process_activity()
        if activity:
            print(f"✅ Categorized: {activity.category} - {activity.subcategory}")
            
            # Show recent summary
            summary = get_activity_summary(1)
            print(f"📊 Last hour: {summary['total_activities']} activities")
        else:
            print("❌ No new activity found")
            
    except Exception as e:
        print(f"❌ Error in activity categorization: {e}")

def run_task_inference():
    """Run task inference"""
    print("\n🎯 Running Task Inference...")
    try:
        from task_inference_agent import analyze_and_infer_tasks
        
        tasks, suggestions = analyze_and_infer_tasks()
        
        if tasks:
            print(f"✅ Generated {len(tasks)} task suggestions")
            for task in tasks[:3]:  # Show top 3
                print(f"   • {task.title}")
        
        if suggestions:
            print(f"💡 {len(suggestions)} productivity suggestions")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
                
    except Exception as e:
        print(f"❌ Error in task inference: {e}")

def run_enhanced_todo():
    """Run enhanced todo agent"""
    print("\n📋 Running Enhanced Todo Agent...")
    try:
        from enhanced_todo_agent import interactive_task_management
        interactive_task_management()
    except Exception as e:
        print(f"❌ Error in enhanced todo agent: {e}")

def run_continuous_monitoring():
    """Run continuous monitoring in a loop"""
    print("\n🔄 Starting Continuous Monitoring Loop...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        from activity_categorizer import process_activity, get_activity_summary
        from task_inference_agent import analyze_and_infer_tasks, load_inferred_tasks
        
        last_activity_time = dt.datetime.now()
        last_inference_time = dt.datetime.now()
        last_summary_time = dt.datetime.now()
        
        # Configuration (in milliseconds)
        ACTIVITY_INTERVAL = 2000 # Process activity every 5 seconds
        INFERENCE_INTERVAL = 2000  # Run inference every 15 seconds
        SUMMARY_INTERVAL = 30000  # Show summary every 30 seconds
        
        print(f"📊 Monitoring Configuration:")
        print(f"   Activity processing: Every {ACTIVITY_INTERVAL/1000:.1f} seconds")
        print(f"   Task inference: Every {INFERENCE_INTERVAL/1000:.1f} seconds")
        print(f"   Summary display: Every {SUMMARY_INTERVAL/1000:.1f} seconds")
        print("=" * 60)
        
        while True:
            current_time = dt.datetime.now()
            
            # Process activity
            if (current_time - last_activity_time).total_seconds() * 1000 >= ACTIVITY_INTERVAL:
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Processing activity...")
                try:
                    activity = process_activity()
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
                    tasks, suggestions = analyze_and_infer_tasks()
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
    print("\n🔄 Starting Background Monitoring...")
    print("Running silently in background. Check logs for activity.")
    print("Press Ctrl+C to stop")
    
    try:
        from activity_categorizer import process_activity
        from task_inference_agent import analyze_and_infer_tasks
        
        last_activity_time = dt.datetime.now()
        last_inference_time = dt.datetime.now()
        
        # Configuration for background mode (in milliseconds)
        ACTIVITY_INTERVAL = 10000  # Process activity every 10 seconds
        INFERENCE_INTERVAL = 30000  # Run inference every 30 seconds
        
        while True:
            current_time = dt.datetime.now()
            
            # Process activity silently
            if (current_time - last_activity_time).total_seconds() * 1000 >= ACTIVITY_INTERVAL:
                try:
                    activity = process_activity()
                    if activity:
                        print(f"[{current_time.strftime('%H:%M:%S')}] Activity: {activity.category} - {activity.subcategory}")
                except Exception as e:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Activity error: {e}")
                last_activity_time = current_time
            
            # Run task inference silently
            if (current_time - last_inference_time).total_seconds() * 1000 >= INFERENCE_INTERVAL:
                try:
                    tasks, suggestions = analyze_and_infer_tasks()
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
    """Run the complete system"""
    print("\n🚀 Running Full Nudge System...")
    
    # Step 1: Activity categorization
    run_activity_categorization()
    
    # Step 2: Task inference
    run_task_inference()
    
    # Step 3: Enhanced todo management
    print("\n" + "="*60)
    print("🎯 READY FOR INTERACTIVE MANAGEMENT")
    print("="*60)
    run_enhanced_todo()

def show_system_status():
    """Show current system status"""
    print("\n📊 System Status:")
    
    # Check data files
    files = {
        "captures/": "Screen captures",
        "user_activities.json": "Activity data", 
        "inferred_tasks.json": "Inferred tasks",
        "tasks.json": "Todo tasks"
    }
    
    for file_path, description in files.items():
        if Path(file_path).exists():
            if Path(file_path).is_dir():
                count = len(list(Path(file_path).glob("*")))
                print(f"✅ {description}: {count} files")
            else:
                print(f"✅ {description}: Found")
        else:
            print(f"❌ {description}: Not found")

def main():
    """Main function"""
    print("🎯 NUDGE SYSTEM")
    print("=" * 40)
    
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
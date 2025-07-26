"""
todo_agent.py
─────────────
Dead‑simple to‑do agent using OpenAI for task analysis and matching.

• Reads ./captures/*.txt and *.png files.
• Analyzes images using OpenAI Vision API.
• Uses OpenAI to match new content against existing tasks.
• Updates existing tasks or creates new ones based on AI analysis.

Run:
    export OPENAI_API_KEY=sk-...
    pip install openai
    python todo_agent.py
"""

import json
import os
import glob
import datetime as dt
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import openai

# ─── Config ────────────────────────────────────────────────────────────────────
CAP_DIR = Path("captures")
TASKS_JSON = Path("tasks.json")

# ─── Helpers ───────────────────────────────────────────────────────────────────
def load_latest_captures() -> Dict[str, Optional[str]]:
    """Load latest text and image captures"""
    txt_files = sorted(CAP_DIR.glob("*.txt"), reverse=True)
    png_files = sorted(CAP_DIR.glob("*.png"), reverse=True)
    json_files = sorted(CAP_DIR.glob("*.json"), reverse=True)
    
    result = {"text": None, "image_path": None, "image_description": None, "metadata": None}
    
    if txt_files:
        result["text"] = txt_files[0].read_text()
    
    if png_files:
        result["image_path"] = str(png_files[0])
        
        # Try to find corresponding metadata file
        png_name = png_files[0].stem
        metadata_file = CAP_DIR / f"{png_name}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    result["metadata"] = json.load(f)
            except Exception as e:
                print(f"Error reading metadata: {e}")
        
        # Analyze the image with context from metadata
        result["image_description"] = analyze_image(png_files[0], result["metadata"])
    
    return result

def analyze_image(image_path: Path, metadata: Optional[Dict] = None) -> str:
    """Analyze image using OpenAI Vision API with context from metadata"""
    try:
        client = openai.OpenAI()
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Build context from metadata
        context = ""
        if metadata:
            context = f"\n\nContext: This screenshot was taken from the application '{metadata.get('appName', 'Unknown')}'"
            if metadata.get('windowTitle'):
                context += f" with window title '{metadata['windowTitle']}'"
            context += f" at {metadata.get('timestamp', 'unknown time')}."
        
        prompt = f"Analyze this screenshot and extract any actionable tasks, to-dos, reminders, or important information that might be relevant for task management. Focus on identifying specific actions a user might need to take.{context}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return f"Image analysis failed: {str(e)}"

def load_tasks():
    if TASKS_JSON.exists():
        with open(TASKS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"tasks": []}

def save_tasks(data):
    with open(TASKS_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

def find_matching_task(content: str, tasks_data: Dict) -> Optional[Dict]:
    """Use OpenAI to find if new content matches existing tasks"""
    if not tasks_data["tasks"]:
        return None
    
    try:
        client = openai.OpenAI()
        
        # Format existing tasks for comparison
        existing_tasks = []
        for task in tasks_data["tasks"]:
            existing_tasks.append(f"ID: {task['id']} - {task['intent']}")
        
        tasks_list = "\n".join(existing_tasks)
        
        prompt = f"""
You are analyzing new content to see if it matches any existing tasks.

NEW CONTENT:
{content}

EXISTING TASKS:
{tasks_list}

Does the new content describe a task that is essentially the same as any existing task? 
If yes, respond with just the task ID (e.g., "task_3"). 
If no match exists, respond with "NO_MATCH".

Consider tasks as matching if they refer to the same general action or goal, even if wording differs.
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        if result == "NO_MATCH":
            return None
        
        # Find the matching task
        for task in tasks_data["tasks"]:
            if task["id"] == result:
                return task
        
        return None
        
    except Exception as e:
        print(f"Error finding matching task: {e}")
        return None

# ─── Main flow ────────────────────────────────────────────────────────────────
def process_content(content: str, content_type: str) -> Tuple[Optional[dict], str]:
    """Process text or image content and return matched task and action taken"""
    tasks_data = load_tasks()
    
    # Try to find matching task using OpenAI
    matched_task = find_matching_task(content, tasks_data)

    if matched_task:
        matched_task["last_seen"] = dt.datetime.utcnow().isoformat()
        action = f"Updated existing task: {matched_task['intent']}"
    else:
        new_id = f"task_{len(tasks_data['tasks'])+1}"
        new_task = {
            "id": new_id,
            "intent": content[:200],  # Increased from 100 to capture more context
            "content_type": content_type,
            "created_at": dt.datetime.utcnow().isoformat(),
            "last_seen": dt.datetime.utcnow().isoformat()
        }
        tasks_data["tasks"].append(new_task)
        action = f"Added new {content_type} task: {new_task['intent'][:60]}…"

    save_tasks(tasks_data)
    return matched_task, action

def main():
    captures = load_latest_captures()
    
    if not captures["text"] and not captures["image_description"]:
        print("No capture content found.")
        return
    
    print(f"Processing captures from {dt.datetime.now().isoformat()}")
    
    # Process text content
    if captures["text"]:
        print("\n--- Processing Text Content ---")
        matched_task, action = process_content(captures["text"], "text")
        print(action)
    
    # Process image content
    if captures["image_description"]:
        print(f"\n--- Processing Image Content ---")
        print(f"Image: {captures['image_path']}")
        print(f"Analysis: {captures['image_description']}")
        matched_task, action = process_content(captures["image_description"], "image")
        print(action)

if __name__ == "__main__":
    main()

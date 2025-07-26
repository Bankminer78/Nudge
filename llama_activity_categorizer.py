"""
llama_activity_categorizer.py
───────────────────────────
LlamaIndex-based activity categorizer with advanced document processing.
Uses vector storage and retrieval for better activity understanding and categorization.

Features:
• Vector-based activity storage and retrieval
• Advanced document processing with LlamaIndex
• Semantic similarity for activity matching
• Rich context extraction and analysis
"""

import json
import os
import datetime as dt
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

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
import chromadb
from pydantic import BaseModel

# Screenshot capture libraries
try:
    import pyautogui
    from PIL import Image, ImageGrab
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False
    print("⚠️  Screenshot libraries not available. Install with: pip install pyautogui pillow")

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
    metadata: Optional[Dict[str, Any]] = None

class ActivityDocument(BaseModel):
    """Pydantic model for activity documents"""
    timestamp: str
    category: str
    subcategory: str
    description: str
    text_content: str
    image_description: Optional[str] = None
    confidence: float
    metadata: Dict[str, Any] = {}

# ─── Config ───────────────────────────────────────────────────────────────────
CAP_DIR = Path("captures")
ACTIVITIES_JSON = Path("user_activities.json")
VECTOR_DB_DIR = Path("vector_db")
CHROMA_DB_DIR = Path("chroma_db")

# Screenshot settings
SCREENSHOT_QUALITY = 85  # JPEG quality (1-100)
SCREENSHOT_FORMAT = "JPEG"  # JPEG or PNG
SAVE_SCREENSHOTS = True  # Whether to save screenshots to disk

# ─── Screenshot Capture ───────────────────────────────────────────────────────
def capture_screenshot() -> Optional[Tuple[bytes, str]]:
    """Capture a screenshot and return image data and filename"""
    if not SCREENSHOT_AVAILABLE:
        print("❌ Screenshot capture not available. Install pyautogui and pillow.")
        return None
    
    try:
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        
        if SCREENSHOT_FORMAT == "JPEG":
            screenshot.save(img_buffer, format='JPEG', quality=SCREENSHOT_QUALITY)
            extension = "jpg"
        else:
            screenshot.save(img_buffer, format='PNG')
            extension = "png"
        
        img_data = img_buffer.getvalue()
        
        # Generate filename
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.{extension}"
        
        # Save to disk if enabled
        if SAVE_SCREENSHOTS:
            CAP_DIR.mkdir(exist_ok=True)
            filepath = CAP_DIR / filename
            with open(filepath, "wb") as f:
                f.write(img_data)
            print(f"📸 Screenshot saved: {filepath}")
        
        return img_data, filename
        
    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        return None

def capture_screenshot_with_ocr() -> Optional[Dict[str, Any]]:
    """Capture screenshot and extract OCR text"""
    if not SCREENSHOT_AVAILABLE:
        print("❌ Screenshot capture not available. Install pyautogui and pillow.")
        return None
    
    try:
        # Capture screenshot
        screenshot_result = capture_screenshot()
        if not screenshot_result:
            return None
        
        img_data, filename = screenshot_result
        
        # Extract OCR text using pyautogui
        try:
            # Use pyautogui's OCR capability
            screenshot = pyautogui.screenshot()
            ocr_text = pyautogui.image_to_string(screenshot)
            
            # Clean up OCR text
            ocr_text = ocr_text.strip()
            if not ocr_text:
                ocr_text = "No text detected in screenshot"
                
        except Exception as ocr_error:
            print(f"⚠️  OCR failed: {ocr_error}")
            ocr_text = "OCR text extraction failed"
        
        return {
            "image_data": img_data,
            "filename": filename,
            "ocr_text": ocr_text,
            "timestamp": dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in screenshot with OCR: {e}")
        return None

def get_screen_info() -> Dict[str, Any]:
    """Get screen information for better context"""
    if not SCREENSHOT_AVAILABLE:
        return {"error": "Screenshot libraries not available"}
    
    try:
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        
        # Get mouse position
        mouse_x, mouse_y = pyautogui.position()
        
        # Get active window info (if available)
        try:
            active_window = pyautogui.getActiveWindow()
            window_title = active_window.title if active_window else "Unknown"
        except:
            window_title = "Unknown"
        
        return {
            "screen_size": f"{screen_width}x{screen_height}",
            "mouse_position": f"({mouse_x}, {mouse_y})",
            "active_window": window_title,
            "timestamp": dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Failed to get screen info: {e}"}

# ─── LlamaIndex Setup ─────────────────────────────────────────────────────────
def setup_llama_index():
    """Initialize LlamaIndex with OpenAI components"""
    try:
        # Initialize OpenAI components
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
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

def get_vector_store():
    """Get or create vector store for activities"""
    try:
        # Create directories if they don't exist
        VECTOR_DB_DIR.mkdir(exist_ok=True)
        CHROMA_DB_DIR.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        chroma_collection = chroma_client.get_or_create_collection("activities")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def get_activity_index():
    """Get or create activity index"""
    try:
        vector_store = get_vector_store()
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
        print(f"Error getting activity index: {e}")
        return None

# ─── Activity Processing ──────────────────────────────────────────────────────
def analyze_image_with_llama(image_data: bytes) -> str:
    """Analyze image using LlamaIndex with OpenAI Vision (image only)"""
    try:
        llm, _ = setup_llama_index()
        if not llm:
            return "Image analysis failed: LlamaIndex not initialized"
        
        # Convert image data to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Get screen context for additional metadata (not sent to OpenAI)
        screen_info = get_screen_info()
        
        # Use LlamaIndex for analysis with image only
        response = llm.complete(
            f"""Analyze this screenshot and provide a rich, detailed description of what the user is doing.

Focus on:
1. **Specific Context**: If on LinkedIn, describe the profile (job title, company, industry, experience level)
2. **Search Intent**: If searching, describe what they're looking for (apartment features, job requirements, etc.)
3. **Content Details**: If reading/watching, describe the topic, source, and relevance
4. **Work Context**: If coding/working, describe the project, technology, or task
5. **Shopping Intent**: If shopping, describe the product category, price range, or specific needs
6. **Application Context**: Consider the application interface and user interaction patterns

Be specific and actionable - this helps with task inference.

Image: {image_base64}"""
        )
        
        return response.text
    except Exception as e:
        print(f"Error analyzing image with LlamaIndex: {e}")
        return f"Image analysis failed: {str(e)}"

def analyze_image_from_path(image_path: Path) -> str:
    """Analyze image from file path (backward compatibility)"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        return analyze_image_with_llama(image_data)
    except Exception as e:
        print(f"Error reading image from path: {e}")
        return f"Image analysis failed: {str(e)}"

def categorize_activity_with_llama(image_description: str) -> Tuple[str, str, float]:
    """Categorize activity using LlamaIndex (image description only)"""
    try:
        llm, _ = setup_llama_index()
        if not llm:
            return fallback_categorization("")
        
        # Use LlamaIndex for categorization with image description only
        prompt = f"""
Analyze this user activity and categorize it with rich context:

Image Analysis: {image_description}

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
        
        response = llm.complete(prompt)
        
        # Parse JSON response
        try:
            result = json.loads(response.text.strip())
            return result["category"], result["subcategory"], result["confidence"]
        except json.JSONDecodeError:
            print(f"JSON parsing error: {response.text}")
            return fallback_categorization("")
        
    except Exception as e:
        print(f"Error categorizing activity with LlamaIndex: {e}")
        return fallback_categorization("")

def fallback_categorization(text_content: str) -> Tuple[str, str, float]:
    """Fallback categorization using pattern matching"""
    text_lower = text_content.lower()
    
    # Activity patterns
    patterns = {
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
    
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            if pattern in text_lower:
                return category.value, pattern, 0.7
    
    return ActivityCategory.UNKNOWN.value, "unknown", 0.5

# ─── Vector Storage Operations ────────────────────────────────────────────────
def store_activity_in_vector_db(activity: UserActivity):
    """Store activity in vector database"""
    try:
        index = get_activity_index()
        if not index:
            return False
        
        # Create document for vector storage
        doc_text = f"""
Activity: {activity.category} - {activity.subcategory}
Description: {activity.description}
Text Content: {activity.text_content}
Timestamp: {activity.timestamp}
Confidence: {activity.confidence}
"""
        
        doc = Document(
            text=doc_text,
            metadata={
                "timestamp": activity.timestamp,
                "category": activity.category,
                "subcategory": activity.subcategory,
                "confidence": activity.confidence,
                "duration_seconds": activity.duration_seconds,
                "image_path": activity.image_path or "",
                **(activity.metadata or {})
            }
        )
        
        # Insert into index
        index.insert(doc)
        
        return True
    except Exception as e:
        print(f"Error storing activity in vector DB: {e}")
        return False

def search_similar_activities(query: str, limit: int = 5) -> List[Dict]:
    """Search for similar activities using vector similarity"""
    try:
        index = get_activity_index()
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
        print(f"Error searching similar activities: {e}")
        return []

# ─── Main Processing ──────────────────────────────────────────────────────────
def load_latest_captures() -> Dict[str, Optional[str]]:
    """Load latest text and image captures (backward compatibility)"""
    txt_files = sorted(CAP_DIR.glob("*.txt"), reverse=True)
    image_files = sorted(CAP_DIR.glob("*.png"), reverse=True) + sorted(CAP_DIR.glob("*.jpg"), reverse=True) + sorted(CAP_DIR.glob("*.jpeg"), reverse=True)
    
    result = {"text": None, "image_path": None, "image_description": None}
    
    if txt_files:
        result["text"] = txt_files[0].read_text()
    
    if image_files:
        result["image_path"] = str(image_files[0])
        result["image_description"] = analyze_image_from_path(image_files[0])
    
    return result

def capture_and_analyze_screenshot() -> Dict[str, Optional[str]]:
    """Capture screenshot directly and analyze it"""
    print("📸 Capturing screenshot...")
    
    # Capture screenshot with OCR (for metadata only)
    screenshot_data = capture_screenshot_with_ocr()
    if not screenshot_data:
        return {"text": None, "image_path": None, "image_description": None}
    
    # Analyze the screenshot (image only, no OCR text)
    print("🔍 Analyzing screenshot with LlamaIndex (image only)...")
    image_description = analyze_image_with_llama(screenshot_data["image_data"])
    
    return {
        "text": screenshot_data["ocr_text"],  # Keep OCR for metadata
        "image_path": screenshot_data["filename"],
        "image_description": image_description,
        "screenshot_data": screenshot_data
    }

def process_activity_with_llama() -> Optional[UserActivity]:
    """Process latest capture and categorize activity using LlamaIndex"""
    # Use direct screenshot capture instead of loading from files
    captures = capture_and_analyze_screenshot()
    
    if not captures["image_description"]:
        print("❌ Failed to capture or analyze screenshot")
        return None
    
    # Use only image description for categorization
    image_description = captures["image_description"]
    
    # Get screen context for additional metadata
    screen_info = get_screen_info()
    
    # Categorize the activity using LlamaIndex (image description only)
    category, subcategory, confidence = categorize_activity_with_llama(image_description)
    
    # Create activity record
    activity = UserActivity(
        timestamp=dt.datetime.utcnow().isoformat(),
        category=category,
        subcategory=subcategory,
        description=image_description[:100] + "..." if len(image_description) > 100 else image_description,
        confidence=confidence,
        text_content=captures["text"] or "",  # Keep OCR for metadata only
        image_path=captures["image_path"],
        metadata={
            "image_description": image_description,
            "processing_method": "llama_index_direct_capture_image_only",
            "screen_info": screen_info,
            "ocr_text": captures["text"] or "",  # Store OCR but not used for analysis
            "screenshot_filename": captures.get("screenshot_data", {}).get("filename", "")
        }
    )
    
    # Store in vector database
    store_activity_in_vector_db(activity)
    
    # Also store in JSON for backward compatibility
    activities = load_activities()
    activities.append(asdict(activity))
    
    # Calculate duration
    if len(activities) > 1:
        activities[-1]["duration_seconds"] = calculate_duration(activities)
    
    # Keep only last 1000 activities
    if len(activities) > 1000:
        activities = activities[-1000:]
    
    save_activities(activities)
    return activity

def process_activity_from_files() -> Optional[UserActivity]:
    """Process activity from existing files (backward compatibility)"""
    captures = load_latest_captures()
    
    if not captures["image_description"]:
        return None
    
    # Use only image description for categorization
    image_description = captures["image_description"]
    
    # Categorize the activity using LlamaIndex (image description only)
    category, subcategory, confidence = categorize_activity_with_llama(image_description)
    
    # Create activity record
    activity = UserActivity(
        timestamp=dt.datetime.utcnow().isoformat(),
        category=category,
        subcategory=subcategory,
        description=image_description[:100] + "..." if len(image_description) > 100 else image_description,
        confidence=confidence,
        text_content=captures["text"] or "",  # Keep text for metadata only
        image_path=captures["image_path"],
        metadata={
            "image_description": image_description,
            "processing_method": "llama_index_file_based_image_only"
        }
    )
    
    # Store in vector database
    store_activity_in_vector_db(activity)
    
    # Also store in JSON for backward compatibility
    activities = load_activities()
    activities.append(asdict(activity))
    
    # Calculate duration
    if len(activities) > 1:
        activities[-1]["duration_seconds"] = calculate_duration(activities)
    
    # Keep only last 1000 activities
    if len(activities) > 1000:
        activities = activities[-1000:]
    
    save_activities(activities)
    return activity

# ─── Backward Compatibility ───────────────────────────────────────────────────
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
    """Calculate duration since last activity"""
    if len(activities) < 2:
        return 60
    
    last_time = dt.datetime.fromisoformat(activities[-1]["timestamp"])
    prev_time = dt.datetime.fromisoformat(activities[-2]["timestamp"])
    duration = (last_time - prev_time).total_seconds()
    
    return max(60, int(duration))

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
    print(f"🦙 Processing activity with LlamaIndex at {dt.datetime.now().isoformat()}")
    
    # Check screenshot capabilities
    if not SCREENSHOT_AVAILABLE:
        print("⚠️  Screenshot capture not available. Install with: pip install pyautogui pillow")
        print("   Falling back to file-based processing...")
        use_direct_capture = False
    else:
        print("✅ Screenshot capture available")
        use_direct_capture = True
    
    # Initialize LlamaIndex
    llm, embed_model = setup_llama_index()
    if not llm or not embed_model:
        print("❌ Failed to initialize LlamaIndex")
        return
    
    print("✅ LlamaIndex initialized successfully")
    
    # Process activity based on available capabilities
    if use_direct_capture:
        print("📸 Using direct screenshot capture (image only)...")
        activity = process_activity_with_llama()
    else:
        print("📁 Using file-based processing (image only)...")
        activity = process_activity_from_files()
    
    if activity:
        print(f"✅ Categorized activity: {activity.category} - {activity.subcategory}")
        print(f"   Description: {activity.description}")
        print(f"   Confidence: {activity.confidence:.2f}")
        print(f"   Processing: {activity.metadata.get('processing_method', 'unknown')}")
        
        # Show screen context if available
        screen_info = activity.metadata.get('screen_info', {})
        if screen_info and 'active_window' in screen_info:
            print(f"   Active Window: {screen_info['active_window']}")
        
        # Show recent summary
        summary = get_activity_summary(1)
        print(f"\n📊 Last hour: {summary['total_activities']} activities")
        
        for category, data in summary["categories"].items():
            duration_minutes = data["total_duration"] // 60
            print(f"   {category}: {data['count']} activities ({duration_minutes} min)")
    else:
        print("❌ No new activity found")

def test_screenshot_capture():
    """Test screenshot capture functionality"""
    print("🧪 Testing screenshot capture...")
    
    if not SCREENSHOT_AVAILABLE:
        print("❌ Screenshot capture not available")
        return
    
    # Test basic screenshot
    print("📸 Testing basic screenshot...")
    screenshot_result = capture_screenshot()
    if screenshot_result:
        print("✅ Basic screenshot successful")
    else:
        print("❌ Basic screenshot failed")
    
    # Test screenshot with OCR (for metadata only)
    print("📸 Testing screenshot with OCR (metadata only)...")
    ocr_result = capture_screenshot_with_ocr()
    if ocr_result:
        print("✅ Screenshot with OCR successful")
        print(f"   OCR Text (metadata): {ocr_result['ocr_text'][:100]}...")
        print("   Note: OCR text is stored for metadata but not used in analysis")
    else:
        print("❌ Screenshot with OCR failed")
    
    # Test screen info
    print("📊 Testing screen info...")
    screen_info = get_screen_info()
    if 'error' not in screen_info:
        print("✅ Screen info successful")
        print(f"   Active Window: {screen_info.get('active_window', 'Unknown')}")
        print(f"   Screen Size: {screen_info.get('screen_size', 'Unknown')}")
    else:
        print("❌ Screen info failed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_screenshot_capture()
    else:
        main() 
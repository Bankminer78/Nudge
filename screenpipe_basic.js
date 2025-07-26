import { pipe } from "@screenpipe/js";
import fs from "fs";
import path from "path";

const CAPTURES_DIR = "captures";
const POLL_INTERVAL = 3000; // 3 seconds
let lastQueryTime = new Date(Date.now() - 5 * 60 * 1000); // Start 5 minutes ago

if (!fs.existsSync(CAPTURES_DIR)) {
  fs.mkdirSync(CAPTURES_DIR, { recursive: true });
}

async function queryScreenpipe() {
  try {
    const now = new Date();
    
    const results = await pipe.queryScreenpipe({
      contentType: "ocr",
      startTime: lastQueryTime.toISOString(),
      endTime: now.toISOString(),
      limit: 50,
      includeFrames: true, // include base64 encoded images
    });

    if (!results || !results.data || results.data.length === 0) {
      console.log(`[${new Date().toISOString()}] No new results found`);
      lastQueryTime = now;
      return;
    }

    console.log(`[${new Date().toISOString()}] Found ${results.data.length} new items`);

    for (const item of results.data) {
      try {
        const timestamp = new Date(item.content.timestamp).toISOString().replace(/[-:T.]/g, '').slice(0, 17);
        
        console.log(`Processing item at ${item.content.timestamp}`);
        console.log(`Text: ${item.content.text}`);
        
        // Save text content
        if (item.content.text && item.content.text.trim()) {
          const textFile = path.join(CAPTURES_DIR, `${timestamp}.txt`);
          fs.writeFileSync(textFile, item.content.text);
        }
        
        // Save image frame if available
        if (item.content.frame) {
          try {
            const imageFile = path.join(CAPTURES_DIR, `${timestamp}.png`);
            const imageBuffer = Buffer.from(item.content.frame, 'base64');
            fs.writeFileSync(imageFile, imageBuffer);
            console.log(`Saved image: ${imageFile}`);
          } catch (imageError) {
            console.warn(`[${new Date().toISOString()}] Failed to save image for ${timestamp}:`, imageError.message);
          }
        }
      } catch (itemError) {
        console.warn(`[${new Date().toISOString()}] Failed to process item:`, itemError.message);
        continue; // Skip this item and continue with the next one
      }
    }

    lastQueryTime = now;
  } catch (error) {
    // Check if it's a video file corruption error (common with screenpipe)
    if (error.message && error.message.includes('moov atom not found')) {
      console.warn(`[${new Date().toISOString()}] Video file corruption detected, skipping this cycle`);
    } else if (error.message && error.message.includes('Invalid data found when processing input')) {
      console.warn(`[${new Date().toISOString()}] Invalid video data detected, skipping this cycle`);
    } else {
      console.error(`[${new Date().toISOString()}] Error querying screenpipe:`, error.message);
    }
    
    // Still update lastQueryTime to avoid getting stuck on the same corrupted data
    lastQueryTime = new Date();
  }
}

async function startServer() {
  console.log(`[${new Date().toISOString()}] Starting screenpipe server...`);
  console.log(`Polling interval: ${POLL_INTERVAL}ms`);
  console.log(`Captures directory: ${CAPTURES_DIR}`);
  
  // Initial query
  await queryScreenpipe();
  
  // Set up continuous polling
  setInterval(queryScreenpipe, POLL_INTERVAL);
  
  console.log(`[${new Date().toISOString()}] Server started successfully`);
}

// Graceful shutdown
process.on('SIGINT', () => {
  console.log(`\n[${new Date().toISOString()}] Shutting down server...`);
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log(`\n[${new Date().toISOString()}] Shutting down server...`);
  process.exit(0);
});

startServer().catch(console.error);
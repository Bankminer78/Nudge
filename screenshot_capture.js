import screenshot from 'screenshot-desktop';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const CAPTURES_DIR = "captures";
const SCREENSHOT_INTERVAL = 5000; // 5 seconds
const MAX_SCREENSHOTS = 100; // Limit to prevent disk space issues

if (!fs.existsSync(CAPTURES_DIR)) {
  fs.mkdirSync(CAPTURES_DIR, { recursive: true });
}

async function getFocusedWindowInfo() {
  try {
    // Get the focused window info using AppleScript
    const script = `
      tell application "System Events"
        set frontApp to first application process whose frontmost is true
        set appName to name of frontApp
        set windowTitle to ""
        try
          set windowTitle to title of first window of frontApp
        end try
        return appName & "|" & windowTitle
      end tell
    `;
    
    const { stdout } = await execAsync(`osascript -e '${script}'`);
    const [appName, windowTitle] = stdout.trim().split('|');
    return { appName: appName || 'Unknown', windowTitle: windowTitle || 'Unknown' };
  } catch (error) {
    console.warn('Failed to get focused window info:', error.message);
    return { appName: 'Unknown', windowTitle: 'Unknown' };
  }
}

async function takeScreenshot() {
  try {
    const timestamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 17);
    const windowInfo = await getFocusedWindowInfo();
    
    console.log(`[${new Date().toISOString()}] Taking screenshot - App: ${windowInfo.appName}, Window: ${windowInfo.windowTitle}`);
    
    // Take screenshot of the entire screen (focused window capture is complex)
    const img = await screenshot({ format: 'png' });
    
    // Create filename with timestamp and app info
    const sanitizedAppName = windowInfo.appName.replace(/[^a-zA-Z0-9]/g, '_');
    const filename = `${timestamp}_${sanitizedAppName}.png`;
    const filepath = path.join(CAPTURES_DIR, filename);
    
    // Save screenshot
    fs.writeFileSync(filepath, img);
    
    // Create metadata file with window context
    const metadataFile = path.join(CAPTURES_DIR, `${timestamp}_${sanitizedAppName}.json`);
    const metadata = {
      timestamp: new Date().toISOString(),
      appName: windowInfo.appName,
      windowTitle: windowInfo.windowTitle,
      filename: filename
    };
    fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));
    
    console.log(`Screenshot saved: ${filename}`);
    
    // Clean up old screenshots if we exceed the limit
    await cleanupOldScreenshots();
    
  } catch (error) {
    console.error(`[${new Date().toISOString()}] Failed to take screenshot:`, error.message);
  }
}

async function cleanupOldScreenshots() {
  try {
    const files = fs.readdirSync(CAPTURES_DIR)
      .filter(file => file.endsWith('.png'))
      .map(file => ({
        name: file,
        path: path.join(CAPTURES_DIR, file),
        mtime: fs.statSync(path.join(CAPTURES_DIR, file)).mtime
      }))
      .sort((a, b) => b.mtime - a.mtime);
    
    if (files.length > MAX_SCREENSHOTS) {
      const filesToDelete = files.slice(MAX_SCREENSHOTS);
      for (const file of filesToDelete) {
        fs.unlinkSync(file.path);
        
        // Also delete corresponding metadata file
        const metadataPath = file.path.replace('.png', '.json');
        if (fs.existsSync(metadataPath)) {
          fs.unlinkSync(metadataPath);
        }
      }
      console.log(`Cleaned up ${filesToDelete.length} old screenshots`);
    }
  } catch (error) {
    console.warn('Failed to cleanup old screenshots:', error.message);
  }
}

async function startScreenshotCapture() {
  console.log(`[${new Date().toISOString()}] Starting screenshot capture...`);
  console.log(`Interval: ${SCREENSHOT_INTERVAL}ms`);
  console.log(`Max screenshots: ${MAX_SCREENSHOTS}`);
  console.log(`Captures directory: ${CAPTURES_DIR}`);
  
  // Take initial screenshot
  await takeScreenshot();
  
  // Set up continuous capture
  setInterval(takeScreenshot, SCREENSHOT_INTERVAL);
  
  console.log(`[${new Date().toISOString()}] Screenshot capture started successfully`);
}

// Graceful shutdown
process.on('SIGINT', () => {
  console.log(`\n[${new Date().toISOString()}] Shutting down screenshot capture...`);
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log(`\n[${new Date().toISOString()}] Shutting down screenshot capture...`);
  process.exit(0);
});

startScreenshotCapture().catch(console.error);
/**
 * Directory check middleware
 * This middleware checks for the existence of important directories
 * and logs detailed information about them
 */

// Use dynamic imports for better compatibility
let fs;
let path;

try {
  const { createRequire } = await import('module');
  const require = createRequire(import.meta.url);
  fs = require('fs');
  path = require('path');
} catch (err) {
  console.error('Error importing fs and path:', err.message);
  // Fallback to ESM imports
  fs = await import('fs');
  path = await import('path');
}

// Check if we're in debug mode
const config = useRuntimeConfig();
const DEBUG = config.debugMode === 'true';

/**
 * Debug logger that only logs when DEBUG is true
 * @param {string} message - The message to log
 * @param {...any} args - Additional arguments to log
 */
function debugLog(message, ...args) {
  if (DEBUG) {
    console.log(`[DIRECTORY CHECK] ${message}`, ...args);
  }
}

/**
 * Check if a directory exists and log its contents
 * @param {string} dirPath - The directory path to check
 * @return {boolean} Whether the directory exists
 */
function checkDirectory(dirPath) {
  try {
    if (fs.existsSync(dirPath)) {
      debugLog(`Directory exists: ${dirPath}`);

      try {
        const contents = fs.readdirSync(dirPath);
        debugLog(`Contents of ${dirPath}:`, contents);
        return true;
      } catch (err) {
        debugLog(`Error reading directory ${dirPath}:`, err.message);
      }
    } else {
      debugLog(`Directory does not exist: ${dirPath}`);
    }
  } catch (err) {
    debugLog(`Error checking directory ${dirPath}:`, err.message);
  }

  return false;
}

/**
 * Create a directory if it doesn't exist
 * @param {string} dirPath - The directory path to create
 */
function ensureDirectoryExists(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) {
      debugLog(`Creating directory: ${dirPath}`);
      fs.mkdirSync(dirPath, { recursive: true });
      debugLog(`Successfully created directory: ${dirPath}`);
    }
  } catch (err) {
    debugLog(`Error creating directory ${dirPath}:`, err.message);
  }
}

export default defineEventHandler(async (event) => {
  // Only run this middleware once per server start
  // Use a global variable to track if we've already run
  if (global._directoryCheckRun) {
    return;
  }

  global._directoryCheckRun = true;

  debugLog('Running directory check middleware');
  debugLog('Environment information:');
  debugLog(`- Current working directory: ${process.cwd()}`);
  debugLog(`- NODE_ENV: ${process.env.NODE_ENV}`);
  debugLog(`- VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

  // Check important directories
  const cwd = process.cwd();

  // Check public directory
  const publicDir = path.resolve(cwd, 'public');
  checkDirectory(publicDir);

  // Check server/data directory
  const serverDataDir = path.resolve(cwd, 'server/data');
  checkDirectory(serverDataDir);

  // Check career_roadmaps directories
  const publicCareerRoadmapsDir = path.resolve(cwd, 'public/career_roadmaps');
  const serverCareerRoadmapsDir = path.resolve(cwd, 'server/data/career_roadmaps');
  const staticDataDir = path.resolve(cwd, 'public/static-data');

  checkDirectory(publicCareerRoadmapsDir);
  checkDirectory(serverCareerRoadmapsDir);
  checkDirectory(staticDataDir);

  // Check Vercel-specific directories
  if (process.env.VERCEL_ENV) {
    debugLog('Checking Vercel-specific directories');

    const vercelDirs = [
      '/var/task/public',
      '/var/task/public/career_roadmaps',
      '/var/task/public/static-data',
      '/var/task/server/data',
      '/var/task/server/data/career_roadmaps'
    ];

    for (const dir of vercelDirs) {
      checkDirectory(dir);
    }

    // Try to create the directories if they don't exist
    debugLog('Attempting to create missing directories in Vercel environment');

    for (const dir of vercelDirs) {
      ensureDirectoryExists(dir);
    }
  }

  debugLog('Directory check middleware completed');

  // Continue processing the request
  return;
});

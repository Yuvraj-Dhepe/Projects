/**
 * Server-side plugin to check and create directories
 * This runs during Nuxt initialization
 */

// Use dynamic imports for better compatibility
let fs;
let path;

export default defineNuxtPlugin(async () => {
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
  const config = useRuntimeConfig();
  const DEBUG = config.debugMode === 'true';

  /**
   * Debug logger that only logs when DEBUG is true
   * @param {string} message - The message to log
   * @param {...any} args - Additional arguments to log
   */
  function debugLog(message, ...args) {
    if (DEBUG) {
      console.log(`[DIRECTORY PLUGIN] ${message}`, ...args);
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

  /**
   * Copy a directory recursively
   * @param {string} src - Source directory
   * @param {string} dest - Destination directory
   */
  function copyDirRecursive(src, dest) {
    try {
      // Create destination directory if it doesn't exist
      ensureDirectoryExists(dest);

      // Read source directory
      const entries = fs.readdirSync(src, { withFileTypes: true });

      // Copy each entry
      for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);

        if (entry.isDirectory()) {
          // Recursively copy directory
          copyDirRecursive(srcPath, destPath);
        } else {
          // Copy file
          fs.copyFileSync(srcPath, destPath);
          debugLog(`Copied file: ${srcPath} -> ${destPath}`);
        }
      }

      debugLog(`Copied directory: ${src} -> ${dest}`);
    } catch (err) {
      debugLog(`Error copying directory ${src} to ${dest}:`, err.message);
    }
  }

  // Run directory checks and creation
  debugLog('Running directory check plugin');
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

  const publicCareerRoadmapsExists = checkDirectory(publicCareerRoadmapsDir);
  const serverCareerRoadmapsExists = checkDirectory(serverCareerRoadmapsDir);
  const staticDataExists = checkDirectory(staticDataDir);

  // If we're in Vercel, try to create the directories and copy data
  if (process.env.VERCEL_ENV) {
    debugLog('Running in Vercel environment, ensuring directories exist');

    // Create directories if they don't exist
    try {
      ensureDirectoryExists('/var/task/public/career_roadmaps');
      ensureDirectoryExists('/var/task/public/static-data');
      ensureDirectoryExists('/var/task/server/data/career_roadmaps');
      debugLog('Successfully created Vercel directories');
    } catch (err) {
      debugLog('Error creating Vercel directories:', err.message);
      debugLog('This is expected in some Vercel environments and will be handled by fallback mechanisms');
    }

    // If server/data/career_roadmaps exists but public/career_roadmaps doesn't, copy it
    if (serverCareerRoadmapsExists && !publicCareerRoadmapsExists) {
      debugLog('Copying server/data/career_roadmaps to public/career_roadmaps');
      copyDirRecursive(serverCareerRoadmapsDir, publicCareerRoadmapsDir);
    }

    // If public/career_roadmaps exists but server/data/career_roadmaps doesn't, copy it
    if (publicCareerRoadmapsExists && !serverCareerRoadmapsExists) {
      debugLog('Copying public/career_roadmaps to server/data/career_roadmaps');
      copyDirRecursive(publicCareerRoadmapsDir, serverCareerRoadmapsDir);
    }

    // If either career_roadmaps exists but static-data doesn't, copy it
    if ((publicCareerRoadmapsExists || serverCareerRoadmapsExists) && !staticDataExists) {
      debugLog('Copying career_roadmaps to static-data');
      if (publicCareerRoadmapsExists) {
        copyDirRecursive(publicCareerRoadmapsDir, staticDataDir);
      } else {
        copyDirRecursive(serverCareerRoadmapsDir, staticDataDir);
      }
    }
  }

  debugLog('Directory check plugin completed');
});

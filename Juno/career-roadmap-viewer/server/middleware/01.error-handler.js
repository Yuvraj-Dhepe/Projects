/**
 * Global error handler middleware
 * This middleware logs errors and provides better error handling
 */

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
    console.log(`[ERROR MIDDLEWARE] ${message}`, ...args);
  }
}

export default defineEventHandler(async (event) => {
  // Log the request
  const method = event.node.req.method;
  const url = event.node.req.url;

  debugLog(`${method} ${url}`);

  // Log environment information on startup
  if (url === '/' && method === 'GET') {
    debugLog('Environment information:');
    debugLog(`- Current working directory: ${process.cwd()}`);
    debugLog(`- NODE_ENV: ${process.env.NODE_ENV}`);
    debugLog(`- VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Try to list directories in the current working directory
    try {
      // Use dynamic import for ESM compatibility
      const { createRequire } = await import('module');
      const require = createRequire(import.meta.url);
      const fs = require('fs');
      const path = require('path');

      const cwdContents = fs.readdirSync(process.cwd());
      debugLog('Contents of current working directory:', cwdContents);

      // Check for public directory
      const publicDir = path.resolve(process.cwd(), 'public');
      if (fs.existsSync(publicDir)) {
        const publicContents = fs.readdirSync(publicDir);
        debugLog('Contents of public directory:', publicContents);

        // Check for career_roadmaps directory
        const careerRoadmapsDir = path.resolve(publicDir, 'career_roadmaps');
        const careerRoadmapsExists = fs.existsSync(careerRoadmapsDir);
        if (careerRoadmapsExists) {
          const careerRoadmapsContents = fs.readdirSync(careerRoadmapsDir);
          debugLog('Contents of career_roadmaps directory:', careerRoadmapsContents);
        } else {
          debugLog('career_roadmaps directory does not exist in public');
        }

        // Check for static-data directory
        const staticDataDir = path.resolve(publicDir, 'static-data');
        const staticDataExists = fs.existsSync(staticDataDir);
        if (staticDataExists) {
          const staticDataContents = fs.readdirSync(staticDataDir);
          debugLog('Contents of static-data directory:', staticDataContents);
        } else {
          debugLog('static-data directory does not exist in public');
        }

        // Check for data directory
        const dataDir = path.resolve(publicDir, 'data');
        const dataExists = fs.existsSync(dataDir);
        if (dataExists) {
          const dataContents = fs.readdirSync(dataDir);
          debugLog('Contents of data directory:', dataContents);
        } else {
          debugLog('data directory does not exist in public');
        }

        // Check server/data directory
        const serverDataDir = path.resolve(process.cwd(), 'server/data');
        const serverDataExists = fs.existsSync(serverDataDir);
        if (serverDataExists) {
          const serverDataContents = fs.readdirSync(serverDataDir);
          debugLog('Contents of server/data directory:', serverDataContents);

          // Check for career_roadmaps in server/data
          const serverCareerRoadmapsDir = path.resolve(serverDataDir, 'career_roadmaps');
          if (fs.existsSync(serverCareerRoadmapsDir)) {
            const serverCareerRoadmapsContents = fs.readdirSync(serverCareerRoadmapsDir);
            debugLog('Contents of server/data/career_roadmaps directory:', serverCareerRoadmapsContents);
          } else {
            debugLog('career_roadmaps directory does not exist in server/data');
          }
        } else {
          debugLog('server/data directory does not exist');
        }

        // Log if all important directories are missing
        if (!careerRoadmapsExists && !staticDataExists && !fs.existsSync(path.resolve(serverDataDir, 'career_roadmaps'))) {
          debugLog('WARNING: All career roadmaps directories are missing!');
          debugLog('The application will use fallback mechanisms to serve content.');
        }
      } else {
        debugLog('public directory does not exist');
      }

      // Check Vercel-specific directories
      if (process.env.VERCEL_ENV) {
        debugLog('Checking Vercel-specific directories');
        try {
          if (fs.existsSync('/var/task/public')) {
            debugLog('/var/task/public exists');
            if (fs.existsSync('/var/task/public/career_roadmaps')) {
              debugLog('/var/task/public/career_roadmaps exists');
            } else {
              debugLog('/var/task/public/career_roadmaps does not exist');
            }
          } else {
            debugLog('/var/task/public does not exist');
          }
        } catch (vercelErr) {
          debugLog('Error checking Vercel directories:', vercelErr.message);
        }
      }
    } catch (err) {
      debugLog('Error listing directories:', err.message);
    }
  }

  // Continue processing the request
  return;
});

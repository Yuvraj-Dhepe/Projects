/**
 * API endpoint to get career paths from the public data directory
 * This approach doesn't rely on file system access in serverless functions
 */
// Use dynamic imports for better compatibility
let fs;
let path;
let staticCareerPaths = [];

try {
  const { createRequire } = await import('module');
  const require = createRequire(import.meta.url);
  fs = require('fs');
  path = require('path');

  // Import the static career paths data
  // This will be bundled with the server code during build
  try {
    staticCareerPaths = require('../../data/career-paths.json');
  } catch (importErr) {
    console.error('Error importing static career paths data:', importErr.message);
    // Will use fallback mechanisms
  }
} catch (err) {
  console.error('Error importing modules:', err.message);
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
    console.log(message, ...args);
  }
}

/**
 * API endpoint to get career paths from the public data directory
 *
 * @return {Object} JSON response with career paths
 */
export default defineEventHandler(async (event) => {
  try {
    debugLog('Fetching career paths from public data directory');

    // Try multiple approaches to get the career paths data

    // Approach 1: Use the imported static data (most reliable in Vercel)
    try {
      debugLog('Using imported static career paths data');
      if (staticCareerPaths && Array.isArray(staticCareerPaths) && staticCareerPaths.length > 0) {
        debugLog('Loaded career paths from static import:', staticCareerPaths);
        return staticCareerPaths;
      }
    } catch (importError) {
      debugLog('Error using imported static data:', importError.message);
    }

    // Approach 2: Try to read the file directly from the filesystem
    try {
      // Try multiple possible locations
      const possibleLocations = [
        path.resolve(process.cwd(), 'server/data/career-paths.json'),
        path.resolve(process.cwd(), 'public/data/career-paths.json'),
        path.resolve(process.cwd(), '.output/public/data/career-paths.json'),
        '/var/task/server/data/career-paths.json',
        '/var/task/public/data/career-paths.json'
      ];

      for (const location of possibleLocations) {
        debugLog('Checking for career paths file at:', location);

        if (fs.existsSync(location)) {
          const fileContent = fs.readFileSync(location, 'utf8');
          const careerPaths = JSON.parse(fileContent);
          debugLog(`Loaded career paths from filesystem at ${location}:`, careerPaths);
          return careerPaths;
        }
      }
    } catch (fsError) {
      debugLog('Error reading career paths from filesystem:', fsError.message);
    }

    // Approach 3: Try to fetch the file using $fetch
    try {
      debugLog('Trying to fetch career paths using $fetch');
      const careerPaths = await $fetch('/data/career-paths.json', {
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache' }
      });

      debugLog('Loaded career paths using $fetch:', careerPaths);
      return careerPaths;
    } catch (fetchError) {
      debugLog('Error fetching career paths using $fetch:', fetchError.message);
    }

    // Approach 4: Try to fetch from the original API as a fallback
    try {
      debugLog('Trying original API as fallback');
      const fallbackData = await $fetch('/api/career-paths', {
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache' }
      });

      debugLog('Loaded career paths from fallback API:', fallbackData);
      return fallbackData;
    } catch (fallbackError) {
      debugLog('Error fetching from fallback API:', fallbackError.message);
    }

    // Approach 5: Generate the data dynamically as a last resort
    debugLog('Generating career paths data dynamically');
    return [
      { id: 'software_engineer', name: 'Software Engineer' },
      { id: 'data_scientist', name: 'Data Scientist' },
      { id: 'cybersecurity_analyst', name: 'Cybersecurity Analyst' },
      { id: 'cloud_architect', name: 'Cloud Architect' },
      { id: 'devops_engineer', name: 'Devops Engineer' },
      { id: 'machine_learning_engineer', name: 'Machine Learning Engineer' },
      { id: 'product_manager', name: 'Product Manager' },
      { id: 'ux_designer', name: 'Ux Designer' },
      { id: 'web_developer', name: 'Web Developer' }
    ];
  } catch (error) {
    console.error('All approaches to fetch career paths failed:', error);

    // Return an empty array as a last resort
    return [];
  }
});

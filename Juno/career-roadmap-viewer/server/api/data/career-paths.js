/**
 * API endpoint to get career paths from the public data directory
 * This approach doesn't rely on file system access in serverless functions
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
    
    // In Nuxt 3, we can use $fetch to get data from the public directory
    // This works in both development and production (Vercel) environments
    const careerPaths = await $fetch('/data/career-paths.json', { 
      method: 'GET',
      headers: { 'Cache-Control': 'no-cache' }
    });
    
    debugLog('Loaded career paths from public data directory:', careerPaths);
    
    // Return the career paths
    return careerPaths;
  } catch (error) {
    console.error('Error fetching career paths from public data:', error);
    
    // Try to fetch from the original API as a fallback
    try {
      debugLog('Trying original API as fallback');
      return await $fetch('/api/career-paths', { 
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache' }
      });
    } catch (fallbackError) {
      console.error('Fallback API also failed:', fallbackError);
      return { 
        error: 'Failed to fetch career paths', 
        message: error.message,
        fallbackError: fallbackError.message
      };
    }
  }
});

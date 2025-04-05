import fs from 'fs';
import path from 'path';

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
 * API endpoint to discover available career paths
 *
 * @return {Object} JSON response with career paths
 */
export default defineEventHandler(async (event) => {
  try {
    // Path to career_roadmaps directory in the public folder
    const careerRoadmapsDir = path.resolve(process.cwd(), 'public/career_roadmaps');
    debugLog('Looking for career paths in:', careerRoadmapsDir);

    // Check if directory exists
    if (!fs.existsSync(careerRoadmapsDir)) {
      console.error('Directory does not exist:', careerRoadmapsDir);
      return [];
    }

    // Read all directories in the career_roadmaps folder
    const allItems = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true });
    debugLog('All items in directory:', allItems.map(item => item.name));

    const directories = allItems
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name)
      .filter(name => !['node_modules', '.git'].includes(name));

    debugLog('Filtered directories:', directories);

    // Filter to only include directories that have diagrams/career_path.mmd
    const careerPaths = directories
      .filter(dir => {
        const diagramPath = path.join(careerRoadmapsDir, dir, 'diagrams', 'career_path.mmd');
        const exists = fs.existsSync(diagramPath);
        debugLog(`Checking for diagram in ${dir}:`, diagramPath, exists ? 'FOUND' : 'NOT FOUND');
        return exists;
      })
      .map(dir => {
        // Format the name for display (convert snake_case to Title Case)
        const name = dir
          .split('_')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');

        return {
          id: dir,
          name: name
        };
      });

    debugLog('Final career paths:', careerPaths);

    // Log the response that will be sent to the client
    debugLog('API Response:', JSON.stringify(careerPaths));

    // Force the response to be an array
    return Array.isArray(careerPaths) ? careerPaths : [];
  } catch (error) {
    console.error('Error discovering career paths:', error);
    return { error: 'Failed to discover career paths', message: error.message };
  }
});

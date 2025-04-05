import fs from 'fs';
import path from 'path';
import { getCareerPathsData } from '../data/career-paths-data';

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
 * API endpoint to get career paths from static data file
 *
 * @return {Object} JSON response with career paths
 */
export default defineEventHandler(async (event) => {
  try {
    debugLog('Fetching career paths from static data file');

    // Try multiple possible locations for the static data file
    const possibleLocations = [
      // Vercel serverless function locations
      path.resolve(process.cwd(), 'static-data/career-paths.json'),
      path.resolve(process.cwd(), '_nuxt/static-data/career-paths.json'),
      path.resolve(process.cwd(), 'public/static-data/career-paths.json'),
      path.resolve(process.cwd(), 'public/_nuxt/static-data/career-paths.json'),

      // Local development locations
      path.resolve(process.cwd(), 'public/static-data/career-paths.json'),
      path.resolve(process.cwd(), 'public/_nuxt/static-data/career-paths.json'),

      // Other possible locations
      path.resolve(process.cwd(), '../static-data/career-paths.json'),
      path.resolve(process.cwd(), '../public/static-data/career-paths.json'),
      path.resolve(process.cwd(), '../public/_nuxt/static-data/career-paths.json'),

      // Absolute paths for Vercel
      '/var/task/static-data/career-paths.json',
      '/var/task/_nuxt/static-data/career-paths.json',
      '/var/task/public/static-data/career-paths.json',
      '/var/task/public/_nuxt/static-data/career-paths.json'
    ];

    debugLog('Trying possible locations for static data file:', possibleLocations);

    // Try each location until we find the file
    let staticDataFile = null;
    let fileContent = null;

    for (const location of possibleLocations) {
      try {
        if (fs.existsSync(location)) {
          staticDataFile = location;
          fileContent = fs.readFileSync(location, 'utf-8');
          debugLog('Found and read static data file at:', location);
          break;
        }
      } catch (err) {
        debugLog(`Error checking location ${location}:`, err.message);
      }
    }

    // If we couldn't find the file, try to create a fallback
    if (!staticDataFile) {
      console.error('Could not find static data file in any location');

      // Try to scan the career_roadmaps directory directly as a last resort
      try {
        const careerRoadmapsDir = path.resolve(process.cwd(), 'public/career_roadmaps');
        if (fs.existsSync(careerRoadmapsDir)) {
          debugLog('Attempting to generate career paths from directory scan');

          const directories = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => {
              const name = dirent.name
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

              return { id: dirent.name, name };
            });

          if (directories.length > 0) {
            debugLog(`Generated ${directories.length} career paths from directory scan`);
            return directories;
          }
        }
      } catch (err) {
        debugLog('Error during fallback directory scan:', err.message);
      }

      // Use hardcoded data as a fallback
      debugLog('Using hardcoded career paths data as fallback');
      const hardcodedData = getCareerPathsData();
      return hardcodedData;
    }

    // Parse the file content we already read
    const careerPaths = JSON.parse(fileContent);

    debugLog('Loaded career paths from static data file:', careerPaths);

    // Return the career paths
    return careerPaths;
  } catch (error) {
    console.error('Error fetching career paths from static data:', error);
    return { error: 'Failed to fetch career paths', message: error.message };
  }
});

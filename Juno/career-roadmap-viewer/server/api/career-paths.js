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
    console.log(`[CAREER PATHS API] ${message}`, ...args);
  }
}

// Import the static career paths data
// This will be bundled with the server code during build
let staticCareerPaths = [];
try {
  // Try to import the static data
  const { createRequire } = await import('module');
  const require = createRequire(import.meta.url);
  staticCareerPaths = require('../data/career-paths.json');
  debugLog('Successfully imported static career paths data');
} catch (err) {
  console.error('Error importing static career paths data:', err.message);
  debugLog('Error importing static career paths data:', err.message);
  // Will use fallback mechanisms
}

/**
 * API endpoint to discover available career paths
 * This version uses multiple approaches to find career paths
 * to ensure it works in all environments including Vercel
 *
 * @return {Object} JSON response with career paths
 */
export default defineEventHandler(async () => {
  try {
    debugLog('Starting career paths discovery');
    debugLog('Environment information:');
    debugLog(`- Current working directory: ${process.cwd()}`);
    debugLog(`- NODE_ENV: ${process.env.NODE_ENV}`);
    debugLog(`- VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Try multiple approaches to get career paths

    // Approach 1: Use the imported static data (most reliable in Vercel)
    try {
      debugLog('Approach 1: Using imported static career paths data');
      if (staticCareerPaths && Array.isArray(staticCareerPaths) && staticCareerPaths.length > 0) {
        debugLog('Loaded career paths from static import:', staticCareerPaths);
        return staticCareerPaths;
      }
    } catch (importError) {
      debugLog('Error using imported static data:', importError.message);
    }

    // Approach 2: Try to read from server/data directory
    try {
      const serverDataPath = path.resolve(process.cwd(), 'server/data/career-paths.json');
      debugLog('Approach 2: Checking for career paths file at:', serverDataPath);

      if (fs.existsSync(serverDataPath)) {
        const fileContent = fs.readFileSync(serverDataPath, 'utf8');
        const careerPaths = JSON.parse(fileContent);
        debugLog('Loaded career paths from server/data:', careerPaths);
        return careerPaths;
      }
    } catch (serverDataError) {
      debugLog('Error reading from server/data:', serverDataError.message);
    }

    // Approach 3: Try to read from public/data directory
    try {
      const publicDataPath = path.resolve(process.cwd(), 'public/data/career-paths.json');
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  debugLog('Approach 3: Checking for career paths file at:', publicDataPath);

      if (fs.existsSync(publicDataPath)) {
        const fileContent = fs.readFileSync(publicDataPath, 'utf8');
        const careerPaths = JSON.parse(fileContent);
        debugLog('Loaded career paths from public/data:', careerPaths);
        return careerPaths;
      }
    } catch (publicDataError) {
      debugLog('Error reading from public/data:', publicDataError.message);
    }

    // Approach 4: Try to read from Vercel-specific locations
    try {
      const vercelLocations = [
        '/var/task/server/data/career-paths.json',
        '/var/task/public/data/career-paths.json',
        '/var/task/.output/server/data/career-paths.json',
        '/var/task/.output/public/data/career-paths.json'
      ];

      debugLog('Approach 4: Checking Vercel-specific locations');

      for (const location of vercelLocations) {
        debugLog('Checking:', location);
        if (fs.existsSync(location)) {
          const fileContent = fs.readFileSync(location, 'utf8');
          const careerPaths = JSON.parse(fileContent);
          debugLog(`Loaded career paths from ${location}:`, careerPaths);
          return careerPaths;
        }
      }
    } catch (vercelError) {
      debugLog('Error reading from Vercel locations:', vercelError.message);
    }

    // Approach 5: Try to scan the career_roadmaps directory (original approach)
    try {
      // Try multiple possible locations for the career_roadmaps directory
      const possibleLocations = [
        path.resolve(process.cwd(), 'public/career_roadmaps'),
        path.resolve(process.cwd(), 'server/data/career_roadmaps'),
        path.resolve(process.cwd(), '.output/public/career_roadmaps'),
        path.resolve(process.cwd(), 'public/static-data'),
        '/var/task/public/career_roadmaps',
        '/var/task/server/data/career_roadmaps',
        '/var/task/public/static-data'
      ];

      debugLog('Approach 5: Scanning career_roadmaps directory');

      for (const careerRoadmapsDir of possibleLocations) {
        debugLog('Checking location:', careerRoadmapsDir);

        if (fs.existsSync(careerRoadmapsDir)) {
          debugLog('Found career_roadmaps directory at:', careerRoadmapsDir);

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

          debugLog('Generated career paths from directory scan:', careerPaths);

          if (careerPaths.length > 0) {
            return careerPaths;
          }
        }
      }
    } catch (scanError) {
      debugLog('Error scanning career_roadmaps directory:', scanError.message);
    }

    // Approach 6: Generate hardcoded data as a last resort
    debugLog('Approach 6: Generating hardcoded career paths data');
    const hardcodedPaths = [
      { id: 'software_engineer', name: 'Software Engineer' },
      { id: 'data_scientist', name: 'Data Scientist' },
      { id: 'cybersecurity_analyst', name: 'Cybersecurity Analyst' },
      { id: 'cloud_architect', name: 'Cloud Architect' },
      { id: 'devops_engineer', name: 'DevOps Engineer' },
      { id: 'machine_learning_engineer', name: 'Machine Learning Engineer' },
      { id: 'product_manager', name: 'Product Manager' },
      { id: 'ux_designer', name: 'UX Designer' },
      { id: 'web_developer', name: 'Web Developer' }
    ];

    debugLog('Using hardcoded career paths:', hardcodedPaths);
    return hardcodedPaths;
  } catch (error) {
    console.error('Error discovering career paths:', error);
    debugLog('Fatal error in career paths discovery:', error.message, error.stack);

    // Return an empty array as a last resort
    return [];
  }
});

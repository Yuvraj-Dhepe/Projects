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
    console.log(`[DIAGRAMS API] ${message}`, ...args);
  }
}

/**
 * API endpoint to serve diagram images
 *
 * @param {Object} event - The event object
 * @return {Object} The diagram image file
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path from the URL parameter
    const careerPath = getRouterParam(event, 'path');
    debugLog(`Requested diagram for career path: ${careerPath}`);

    // Get the format from query parameter (default to png)
    const query = getQuery(event);
    const format = query.format || 'png';
    debugLog(`Requested format: ${format}`);

    // Validate format
    if (!['png', 'svg'].includes(format)) {
      throw new Error('Invalid format. Must be png or svg.');
    }

    // Log environment information
    debugLog('Environment information:');
    debugLog(`- Current working directory: ${process.cwd()}`);
    debugLog(`- NODE_ENV: ${process.env.NODE_ENV}`);
    debugLog(`- VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Try multiple possible locations for the career_roadmaps directory
    const possibleLocations = [
      // Standard locations
      path.resolve(process.cwd(), 'public/career_roadmaps'),
      path.resolve(process.cwd(), 'server/data/career_roadmaps'),
      path.resolve(process.cwd(), '.output/public/career_roadmaps'),
      // Vercel-specific locations
      '/var/task/public/career_roadmaps',
      '/var/task/server/data/career_roadmaps',
      // Absolute paths based on cwd
      path.join(process.cwd(), 'public/career_roadmaps'),
      path.join(process.cwd(), 'server/data/career_roadmaps')
    ];

    // Log all possible locations we're checking
    debugLog('Checking possible locations for career_roadmaps directory:');
    possibleLocations.forEach(loc => debugLog(`- ${loc}`));

    // Find the first location that exists
    let careerRoadmapsDir = null;
    for (const location of possibleLocations) {
      try {
        if (fs.existsSync(location)) {
          careerRoadmapsDir = location;
          debugLog(`Found career_roadmaps directory at: ${location}`);
          break;
        }
      } catch (err) {
        debugLog(`Error checking location ${location}:`, err.message);
      }
    }

    if (!careerRoadmapsDir) {
      debugLog('Could not find career_roadmaps directory in any location');

      // Try to list directories in public to see what's available
      try {
        const publicDir = path.resolve(process.cwd(), 'public');
        debugLog(`Checking contents of public directory: ${publicDir}`);
        if (fs.existsSync(publicDir)) {
          const contents = fs.readdirSync(publicDir);
          debugLog('Contents of public directory:', contents);
        } else {
          debugLog('Public directory does not exist');
        }
      } catch (err) {
        debugLog('Error listing public directory:', err.message);
      }

      // Try to list directories in server/data to see what's available
      try {
        const serverDataDir = path.resolve(process.cwd(), 'server/data');
        debugLog(`Checking contents of server/data directory: ${serverDataDir}`);
        if (fs.existsSync(serverDataDir)) {
          const contents = fs.readdirSync(serverDataDir);
          debugLog('Contents of server/data directory:', contents);
        } else {
          debugLog('server/data directory does not exist');
        }
      } catch (err) {
        debugLog('Error listing server/data directory:', err.message);
      }

      throw new Error('Career roadmaps directory not found in any location');
    }

    // Path to the diagram file
    const diagramPath = path.join(careerRoadmapsDir, careerPath, 'diagrams', `career_path.${format}`);
    debugLog(`Looking for diagram at: ${diagramPath}`);

    // Check if the career path directory exists
    const careerPathDir = path.join(careerRoadmapsDir, careerPath);
    if (!fs.existsSync(careerPathDir)) {
      debugLog(`Career path directory does not exist: ${careerPathDir}`);

      // List available career paths
      try {
        const availablePaths = fs.readdirSync(careerRoadmapsDir);
        debugLog('Available career paths:', availablePaths);
      } catch (err) {
        debugLog('Error listing available career paths:', err.message);
      }

      throw new Error(`Career path directory not found: ${careerPathDir}`);
    }

    // Check if the diagrams directory exists
    const diagramsDir = path.join(careerPathDir, 'diagrams');
    if (!fs.existsSync(diagramsDir)) {
      debugLog(`Diagrams directory does not exist: ${diagramsDir}`);

      // List contents of career path directory
      try {
        const contents = fs.readdirSync(careerPathDir);
        debugLog(`Contents of ${careerPath} directory:`, contents);
      } catch (err) {
        debugLog(`Error listing contents of ${careerPath} directory:`, err.message);
      }

      throw new Error(`Diagrams directory not found: ${diagramsDir}`);
    }

    // Check if the file exists
    if (!fs.existsSync(diagramPath)) {
      debugLog(`Diagram file not found: ${diagramPath}`);

      // List available diagram files
      try {
        const availableDiagrams = fs.readdirSync(diagramsDir);
        debugLog('Available diagrams:', availableDiagrams);
      } catch (err) {
        debugLog('Error listing available diagrams:', err.message);
      }

      throw new Error(`Diagram file not found: ${diagramPath}`);
    }

    // Read the file
    debugLog(`Reading diagram file: ${diagramPath}`);
    const fileContent = fs.readFileSync(diagramPath);
    debugLog(`Successfully read diagram file (${fileContent.length} bytes)`);

    // Set appropriate content type
    const contentType = format === 'png' ? 'image/png' : 'image/svg+xml';

    // Set headers for proper caching and content handling
    setResponseHeader(event, 'Content-Type', contentType);
    setResponseHeader(event, 'Cache-Control', 'public, max-age=3600'); // Cache for 1 hour
    setResponseHeader(event, 'Content-Disposition', `inline; filename="career_path.${format}"`); // Help with downloads
    setResponseHeader(event, 'X-Content-Type-Options', 'nosniff'); // Security best practice

    // For downloads, add attachment disposition
    if (query.download === 'true') {
      setResponseHeader(event, 'Content-Disposition', `attachment; filename="${careerPath}_career_path.${format}"`);
    }

    debugLog('Successfully serving diagram');
    // Return the file content
    return fileContent;
  } catch (error) {
    console.error('Error serving diagram:', error);
    debugLog('Error serving diagram:', error.message, error.stack);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Diagram not found', message: error.message };
  }
});

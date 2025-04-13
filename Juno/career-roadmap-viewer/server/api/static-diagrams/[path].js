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
    console.log(`[STATIC DIAGRAMS API] ${message}`, ...args);
  }
}

/**
 * API endpoint to serve diagram images from static data
 *
 * @param {Object} event - The event object
 * @return {Object} The diagram image file
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path from the URL parameter
    const careerPath = getRouterParam(event, 'path');
    debugLog(`Requested static diagram for career path: ${careerPath}`);

    // Get the format from query parameter (default to png)
    const query = getQuery(event);
    const format = query.format || 'png';
    debugLog(`Requested format: ${format}`);

    // Validate format
    if (!['png', 'svg', 'mmd'].includes(format)) {
      throw new Error('Invalid format. Must be png, svg, or mmd.');
    }

    // Log environment information
    debugLog('Environment information:');
    debugLog(`- Current working directory: ${process.cwd()}`);
    debugLog(`- NODE_ENV: ${process.env.NODE_ENV}`);
    debugLog(`- VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Try to list directories in the current working directory
    try {
      const cwdContents = fs.readdirSync(process.cwd());
      debugLog('Contents of current working directory:', cwdContents);
    } catch (err) {
      debugLog('Error listing current working directory:', err.message);
    }

    // Try multiple possible locations for the diagram file
    const possibleLocations = [
      // Server data directory (bundled with the app)
      path.resolve(process.cwd(), `server/data/career_roadmaps/${careerPath}/diagrams/career_path.${format}`),

      // Static data directory (copied during build)
      path.resolve(process.cwd(), `static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `public/static-data/${careerPath}/diagrams/career_path.${format}`),

      // Nuxt static directories
      path.resolve(process.cwd(), `_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),

      // Original source location
      path.resolve(process.cwd(), `public/career_roadmaps/${careerPath}/diagrams/career_path.${format}`),

      // Other possible locations
      path.resolve(process.cwd(), `../static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `../public/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `../public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `../public/career_roadmaps/${careerPath}/diagrams/career_path.${format}`),

      // Absolute paths for Vercel
      `/var/task/server/data/career_roadmaps/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/public/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/public/career_roadmaps/${careerPath}/diagrams/career_path.${format}`
    ];

    debugLog('Checking possible locations for diagram file:');
    possibleLocations.forEach(loc => debugLog(`- ${loc}`));

    // Try each location until we find the file
    let diagramPath = null;
    let fileContent = null;

    for (const location of possibleLocations) {
      try {
        if (fs.existsSync(location)) {
          diagramPath = location;
          fileContent = fs.readFileSync(location);
          debugLog(`Found and read diagram file at: ${location} (${fileContent.length} bytes)`);
          break;
        }
      } catch (err) {
        debugLog(`Error checking location ${location}:`, err.message);
      }
    }

    // If we couldn't find the file, try to list available directories
    if (!diagramPath || !fileContent) {
      debugLog('Could not find diagram file in any location');

      // Try to list static-data directory
      try {
        const staticDataDir = path.resolve(process.cwd(), 'static-data');
        debugLog(`Checking if static-data directory exists: ${staticDataDir}`);
        if (fs.existsSync(staticDataDir)) {
          const contents = fs.readdirSync(staticDataDir);
          debugLog('Contents of static-data directory:', contents);

          // If the career path directory exists, list its contents
          const careerPathDir = path.join(staticDataDir, careerPath);
          if (fs.existsSync(careerPathDir)) {
            const careerPathContents = fs.readdirSync(careerPathDir);
            debugLog(`Contents of ${careerPath} directory:`, careerPathContents);

            // If the diagrams directory exists, list its contents
            const diagramsDir = path.join(careerPathDir, 'diagrams');
            if (fs.existsSync(diagramsDir)) {
              const diagramsContents = fs.readdirSync(diagramsDir);
              debugLog('Contents of diagrams directory:', diagramsContents);
            }
          }
        } else {
          debugLog('static-data directory does not exist');
        }
      } catch (err) {
        debugLog('Error listing static-data directory:', err.message);
      }

      // Try to list server/data directory
      try {
        const serverDataDir = path.resolve(process.cwd(), 'server/data/career_roadmaps');
        debugLog(`Checking if server/data/career_roadmaps directory exists: ${serverDataDir}`);
        if (fs.existsSync(serverDataDir)) {
          const contents = fs.readdirSync(serverDataDir);
          debugLog('Contents of server/data/career_roadmaps directory:', contents);
        } else {
          debugLog('server/data/career_roadmaps directory does not exist');
        }
      } catch (err) {
        debugLog('Error listing server/data directory:', err.message);
      }

      throw new Error(`Diagram file not found for ${careerPath} in format ${format}. Tried ${possibleLocations.length} locations.`);
    }

    // Set appropriate content type
    let contentType;
    switch (format) {
      case 'png':
        contentType = 'image/png';
        break;
      case 'svg':
        contentType = 'image/svg+xml';
        break;
      case 'mmd':
        contentType = 'text/plain';
        break;
    }

    debugLog(`Setting content type to: ${contentType}`);
    setResponseHeader(event, 'Content-Type', contentType);
    setResponseHeader(event, 'Cache-Control', 'public, max-age=3600'); // Cache for 1 hour
    setResponseHeader(event, 'X-Content-Type-Options', 'nosniff'); // Security best practice

    // For downloads, add attachment disposition
    if (query.download === 'true') {
      debugLog('Setting attachment disposition for download');
      setResponseHeader(event, 'Content-Disposition', `attachment; filename="${careerPath}_career_path.${format}"`);
    } else {
      setResponseHeader(event, 'Content-Disposition', `inline; filename="career_path.${format}"`);
    }

    debugLog('Successfully serving static diagram');
    // Return the file content
    return fileContent;
  } catch (error) {
    console.error('Error serving static diagram:', error);
    debugLog('Error serving static diagram:', error.message, error.stack);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Static diagram not found', message: error.message };
  }
});

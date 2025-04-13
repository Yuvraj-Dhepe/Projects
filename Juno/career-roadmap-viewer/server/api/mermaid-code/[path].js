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

// Import static diagrams data
import { getDiagramMmd } from '../../data/static-diagrams-data';

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
    console.log(`[MERMAID CODE API] ${message}`, ...args);
  }
}

/**
 * API endpoint to get the raw Mermaid code
 *
 * @param {Object} event - The event object
 * @return {Object} The Mermaid code
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path from the URL parameter
    const careerPath = getRouterParam(event, 'path');
    debugLog(`Requested Mermaid code for career path: ${careerPath}`);
    debugLog(`Current working directory: ${process.cwd()}`);
    debugLog(`NODE_ENV: ${process.env.NODE_ENV}`);
    debugLog(`VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Try multiple possible locations for the Mermaid file
    const possibleLocations = [
      // Standard locations
      path.resolve(process.cwd(), 'public/career_roadmaps', careerPath, 'diagrams/career_path.mmd'),
      path.resolve(process.cwd(), 'server/data/career_roadmaps', careerPath, 'diagrams/career_path.mmd'),
      path.resolve(process.cwd(), 'public/static-data', careerPath, 'diagrams/career_path.mmd'),
      // Vercel-specific locations
      `/var/task/public/career_roadmaps/${careerPath}/diagrams/career_path.mmd`,
      `/var/task/server/data/career_roadmaps/${careerPath}/diagrams/career_path.mmd`,
      `/var/task/public/static-data/${careerPath}/diagrams/career_path.mmd`,
    ];

    debugLog('Checking possible locations for Mermaid file:');
    possibleLocations.forEach(loc => debugLog(`- ${loc}`));

    // Try each location until we find the file
    for (const mermaidPath of possibleLocations) {
      try {
        if (fs.existsSync(mermaidPath)) {
          debugLog(`Found Mermaid file at: ${mermaidPath}`);
          const mermaidCode = fs.readFileSync(mermaidPath, 'utf-8');
          debugLog(`Successfully read Mermaid file (${mermaidCode.length} bytes)`);
          return { code: mermaidCode };
        }
      } catch (err) {
        debugLog(`Error checking location ${mermaidPath}:`, err.message);
      }
    }

    // If we couldn't find the file, use the static data
    debugLog('Could not find Mermaid file in any location, using static data');
    const staticMermaidCode = getDiagramMmd(careerPath);
    debugLog(`Generated static Mermaid code (${staticMermaidCode.length} bytes)`);

    // Return the static Mermaid code
    return { code: staticMermaidCode };
  } catch (error) {
    console.error('Error getting Mermaid code:', error);
    debugLog('Error getting Mermaid code:', error.message, error.stack);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Mermaid code not found', message: error.message };
  }
});

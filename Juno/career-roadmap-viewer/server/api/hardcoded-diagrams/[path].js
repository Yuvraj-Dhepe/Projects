/**
 * Hardcoded diagrams API endpoint
 * This is a last-resort fallback that returns a simple placeholder
 * when all other approaches fail
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
    console.log(`[HARDCODED DIAGRAMS API] ${message}`, ...args);
  }
}

/**
 * API endpoint to serve hardcoded diagram information
 * This is a fallback for when file system access fails
 *
 * @param {Object} event - The event object
 * @return {Object} The diagram information
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path from the URL parameter
    const careerPath = getRouterParam(event, 'path');
    debugLog(`Requested hardcoded diagram for career path: ${careerPath}`);

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
    setResponseHeader(event, 'Cache-Control', 'public, max-age=60'); // Short cache time for placeholder
    setResponseHeader(event, 'X-Content-Type-Options', 'nosniff');

    // For downloads, add attachment disposition
    if (query.download === 'true') {
      debugLog('Setting attachment disposition for download');
      setResponseHeader(event, 'Content-Disposition', `attachment; filename="${careerPath}_career_path.${format}"`);
    } else {
      setResponseHeader(event, 'Content-Disposition', `inline; filename="career_path.${format}"`);
    }

    // Return a placeholder response based on format
    if (format === 'mmd') {
      // Return a simple mermaid diagram as text
      debugLog('Returning hardcoded mermaid diagram');
      return `graph TD
    A[Education] --> B[Entry Level]
    B --> C[Mid-Level]
    C --> D[Senior Level]
    D --> E[Leadership]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px`;
    } else if (format === 'svg') {
      // Return a simple SVG with the career path name
      debugLog('Returning hardcoded SVG diagram');
      const svgContent = `
        <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
          <rect width="800" height="600" fill="#f0f0f0" />
          <text x="400" y="250" font-family="Arial" font-size="24" text-anchor="middle" fill="#333">
            Career Path: ${careerPath.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
          </text>
          <text x="400" y="300" font-family="Arial" font-size="18" text-anchor="middle" fill="#666">
            (Placeholder diagram - actual diagram not found)
          </text>
          <text x="400" y="350" font-family="Arial" font-size="14" text-anchor="middle" fill="#999">
            Please check server logs for more information
          </text>
        </svg>
      `;
      return svgContent;
    } else {
      // For PNG format, redirect to a placeholder image
      debugLog('Redirecting to placeholder image for PNG format');
      setResponseStatus(event, 307); // Temporary Redirect
      const placeholderText = encodeURIComponent(`Career Path: ${careerPath.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}`);
      setResponseHeader(event, 'Location', `https://via.placeholder.com/800x600?text=${placeholderText}`);
      return null;
    }
  } catch (error) {
    console.error('Error serving hardcoded diagram:', error);
    debugLog('Error serving hardcoded diagram:', error.message, error.stack);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Diagram not found', message: error.message };
  }
});

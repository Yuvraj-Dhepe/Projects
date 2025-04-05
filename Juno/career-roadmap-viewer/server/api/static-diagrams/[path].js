import fs from 'fs';
import path from 'path';

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

    // Get the format from query parameter (default to png)
    const query = getQuery(event);
    const format = query.format || 'png';

    // Validate format
    if (!['png', 'svg', 'mmd'].includes(format)) {
      throw new Error('Invalid format. Must be png, svg, or mmd.');
    }

    // Try multiple possible locations for the diagram file
    const possibleLocations = [
      // Vercel serverless function locations
      path.resolve(process.cwd(), `static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `public/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),

      // Local development locations
      path.resolve(process.cwd(), `public/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),

      // Original source location
      path.resolve(process.cwd(), `public/career_roadmaps/${careerPath}/diagrams/career_path.${format}`),

      // Other possible locations
      path.resolve(process.cwd(), `../static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `../public/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `../public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`),
      path.resolve(process.cwd(), `../public/career_roadmaps/${careerPath}/diagrams/career_path.${format}`),

      // Absolute paths for Vercel
      `/var/task/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/public/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/public/_nuxt/static-data/${careerPath}/diagrams/career_path.${format}`,
      `/var/task/public/career_roadmaps/${careerPath}/diagrams/career_path.${format}`
    ];

    console.log('Trying possible locations for diagram file:', possibleLocations);

    // Try each location until we find the file
    let diagramPath = null;
    let fileContent = null;

    for (const location of possibleLocations) {
      try {
        if (fs.existsSync(location)) {
          diagramPath = location;
          fileContent = fs.readFileSync(location);
          console.log('Found and read diagram file at:', location);
          break;
        }
      } catch (err) {
        console.log(`Error checking location ${location}:`, err.message);
      }
    }

    // If we couldn't find the file, throw an error
    if (!diagramPath || !fileContent) {
      throw new Error(`Diagram file not found for ${careerPath} in format ${format}. Tried ${possibleLocations.length} locations.`);
    }

    // We've already read the file content above

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

    setResponseHeader(event, 'Content-Type', contentType);
    setResponseHeader(event, 'Cache-Control', 'public, max-age=3600'); // Cache for 1 hour

    // For downloads, add attachment disposition
    if (query.download === 'true') {
      setResponseHeader(event, 'Content-Disposition', `attachment; filename="${careerPath}_career_path.${format}"`);
    }

    // Return the file content
    return fileContent;
  } catch (error) {
    console.error('Error serving diagram:', error);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Diagram not found', message: error.message };
  }
});

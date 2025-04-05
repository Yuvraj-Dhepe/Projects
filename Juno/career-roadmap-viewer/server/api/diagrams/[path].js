import fs from 'fs';
import path from 'path';

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

    // Get the format from query parameter (default to png)
    const query = getQuery(event);
    const format = query.format || 'png';

    // Ignore cache-busting parameter if present
    // const timestamp = query.t;

    // Validate format
    if (!['png', 'svg'].includes(format)) {
      throw new Error('Invalid format. Must be png or svg.');
    }

    // Path to the diagram file
    const careerRoadmapsDir = path.resolve(process.cwd(), 'public/career_roadmaps');
    const diagramPath = path.join(careerRoadmapsDir, careerPath, 'diagrams', `career_path.${format}`);

    // Check if the file exists
    if (!fs.existsSync(diagramPath)) {
      throw new Error(`Diagram file not found: ${diagramPath}`);
    }

    // Read the file
    const fileContent = fs.readFileSync(diagramPath);

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

    // Return the file content
    return fileContent;
  } catch (error) {
    console.error('Error serving diagram:', error);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Diagram not found', message: error.message };
  }
});

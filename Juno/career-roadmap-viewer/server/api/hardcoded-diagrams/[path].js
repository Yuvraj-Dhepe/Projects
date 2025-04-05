import { getCareerPathById } from '../../data/career-paths-data';

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

    // Get the format from query parameter (default to png)
    const query = getQuery(event);
    const format = query.format || 'png';
    
    // Validate format
    if (!['png', 'svg', 'mmd'].includes(format)) {
      throw new Error('Invalid format. Must be png, svg, or mmd.');
    }

    // Check if the career path exists in our hardcoded data
    const careerPathData = getCareerPathById(careerPath);
    if (!careerPathData) {
      throw new Error(`Career path not found: ${careerPath}`);
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
    
    setResponseHeader(event, 'Content-Type', contentType);
    
    // For downloads, add attachment disposition
    if (query.download === 'true') {
      setResponseHeader(event, 'Content-Disposition', `attachment; filename="${careerPath}_career_path.${format}"`);
    }

    // Return a placeholder response
    if (format === 'mmd') {
      // Return a simple mermaid diagram as text
      return `graph TD
    A[Education] --> B[Entry Level]
    B --> C[Mid-Level]
    C --> D[Senior Level]
    D --> E[Leadership]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px`;
    } else {
      // For image formats, redirect to a placeholder image
      // Set status to 307 (Temporary Redirect)
      setResponseStatus(event, 307);
      setResponseHeader(event, 'Location', 'https://via.placeholder.com/800x600?text=Career+Path+Diagram');
      return null;
    }
  } catch (error) {
    console.error('Error serving hardcoded diagram:', error);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Diagram not found', message: error.message };
  }
});

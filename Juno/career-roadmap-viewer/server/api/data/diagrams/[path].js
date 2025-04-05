/**
 * API endpoint to serve diagram images from the public data directory
 * This approach doesn't rely on file system access in serverless functions
 */

/**
 * API endpoint to serve diagram images
 *
 * @param {Object} event - The event object
 * @return {Object} The diagram image file or a redirect
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

    // For download requests, we need to set headers and then redirect
    if (query.download === 'true') {
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
      setResponseHeader(event, 'Content-Disposition', `attachment; filename="${careerPath}_career_path.${format}"`);
    }

    // Redirect to the file in the public directory
    // This approach works because Nuxt serves static files from the public directory
    const diagramUrl = `/data/career_roadmaps/${careerPath}/diagrams/career_path.${format}`;
    
    // For direct access (not download), just redirect
    if (query.download !== 'true') {
      // Set status to 307 (Temporary Redirect)
      setResponseStatus(event, 307);
      setResponseHeader(event, 'Location', diagramUrl);
      return null;
    } else {
      // For downloads, we need to fetch the content and return it with the headers we set above
      try {
        // Use $fetch to get the file content
        const response = await $fetch(diagramUrl, { 
          method: 'GET',
          responseType: 'arrayBuffer'
        });
        
        // Return the file content
        return response;
      } catch (fetchError) {
        console.error('Error fetching diagram file:', fetchError);
        throw new Error(`Failed to fetch diagram file: ${fetchError.message}`);
      }
    }
  } catch (error) {
    console.error('Error serving diagram:', error);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Diagram not found', message: error.message };
  }
});

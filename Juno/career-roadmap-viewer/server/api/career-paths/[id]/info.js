import { getCareerPathInfoAsHtml } from '../../../utils/markdownConverter';

export default defineEventHandler(async (event) => {
  try {
    const id = event.context.params.id;
    
    if (!id) {
      return {
        statusCode: 400,
        body: 'Career path ID is required'
      };
    }
    
    // Get the career path info from the markdown file
    const careerPathInfo = getCareerPathInfoAsHtml(id);
    
    if (!careerPathInfo) {
      return {
        statusCode: 404,
        body: `Career path information not found for ID: ${id}`
      };
    }
    
    return {
      html: careerPathInfo
    };
  } catch (error) {
    console.error('Error fetching career path info:', error);
    return {
      statusCode: 500,
      body: 'Error fetching career path information'
    };
  }
});

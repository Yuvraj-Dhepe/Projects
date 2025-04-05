import fs from 'fs';
import path from 'path';
import { marked } from 'marked';

/**
 * API endpoint to get career path information from README.md
 *
 * @param {Object} event - The event object
 * @return {Object} The career path information
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path from the URL parameter
    const careerPath = getRouterParam(event, 'path');

    // Path to the README.md file
    const careerRoadmapsDir = path.resolve(process.cwd(), 'public/career_roadmaps');
    const readmePath = path.join(careerRoadmapsDir, careerPath, 'README.md');

    // Check if the file exists
    if (!fs.existsSync(readmePath)) {
      return { content: null };
    }

    // Read the file
    const fileContent = fs.readFileSync(readmePath, 'utf-8');

    // Convert markdown to HTML
    const htmlContent = marked(fileContent);

    // Return the HTML content
    return { content: htmlContent };
  } catch (error) {
    console.error('Error getting career path info:', error);

    // Set status code and return error
    setResponseStatus(event, 500);
    return { error: 'Failed to get career path info', message: error.message };
  }
});

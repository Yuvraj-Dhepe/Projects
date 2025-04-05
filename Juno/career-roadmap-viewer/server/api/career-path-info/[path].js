import fs from 'fs';
import path from 'path';
import { marked } from 'marked';

/**
 * API endpoint to get career path information from career_goal.md
 *
 * @param {Object} event - The event object
 * @return {Object} The career path information
 */
export default defineEventHandler(async (event) => {
  try {
    // Get the career path from the URL parameter
    const careerPath = getRouterParam(event, 'path');

    // Path to the career_goal.md file
    const careerRoadmapsDir = path.resolve(process.cwd(), 'public/career_roadmaps');
    const careerGoalPath = path.join(careerRoadmapsDir, careerPath, 'career_goal.md');

    // Fallback to README.md if career_goal.md doesn't exist
    const readmePath = path.join(careerRoadmapsDir, careerPath, 'README.md');

    let filePath;

    // Check which file exists and use it
    if (fs.existsSync(careerGoalPath)) {
      filePath = careerGoalPath;
    } else if (fs.existsSync(readmePath)) {
      filePath = readmePath;
    } else {
      return { content: null };
    }

    // Read the file
    const fileContent = fs.readFileSync(filePath, 'utf-8');

    // Convert markdown to HTML with custom styling for career_goal.md
    const htmlContent = marked(fileContent);

    // Add custom styling for better card-based display
    const styledHtmlContent = `
      <div class="career-info">
        ${htmlContent}
      </div>
    `;

    // Return the HTML content
    return { content: styledHtmlContent };
  } catch (error) {
    console.error('Error getting career path info:', error);

    // Set status code and return error
    setResponseStatus(event, 500);
    return { error: 'Failed to get career path info', message: error.message };
  }
});

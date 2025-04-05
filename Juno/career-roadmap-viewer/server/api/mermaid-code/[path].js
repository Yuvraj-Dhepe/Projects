import fs from 'fs';
import path from 'path';

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

    // Path to the Mermaid file
    const careerRoadmapsDir = path.resolve(process.cwd(), 'public/career_roadmaps');
    const mermaidPath = path.join(careerRoadmapsDir, careerPath, 'diagrams', 'career_path.mmd');

    // Check if the file exists
    if (!fs.existsSync(mermaidPath)) {
      throw new Error(`Mermaid file not found: ${mermaidPath}`);
    }

    // Read the file
    const mermaidCode = fs.readFileSync(mermaidPath, 'utf-8');

    // Return the Mermaid code
    return { code: mermaidCode };
  } catch (error) {
    console.error('Error getting Mermaid code:', error);

    // Set status code and return error
    setResponseStatus(event, 404);
    return { error: 'Mermaid code not found', message: error.message };
  }
});

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
    console.log(`[MARKDOWN CONVERTER] ${message}`, ...args);
  }
}
import { marked } from 'marked';

/**
 * Converts markdown content to HTML
 *
 * @param {string} markdown - The markdown content to convert
 * @return {string} - The HTML content
 */
export function convertMarkdownToHtml(markdown) {
  return marked(markdown);
}

/**
 * Reads a markdown file and converts it to HTML
 *
 * @param {string} filePath - Path to the markdown file
 * @return {string} - The HTML content, or empty string if file not found
 */
export function readMarkdownFileAsHtml(filePath) {
  try {
    debugLog(`Reading markdown file: ${filePath}`);
    const markdown = fs.readFileSync(filePath, 'utf-8');
    debugLog(`Successfully read markdown file (${markdown.length} bytes)`);
    return convertMarkdownToHtml(markdown);
  } catch (error) {
    console.error(`Error reading markdown file at ${filePath}:`, error);
    debugLog(`Error reading markdown file: ${error.message}`);
    return '';
  }
}

/**
 * Gets the career path information as HTML from the markdown file
 *
 * @param {string} careerPathId - The ID of the career path
 * @return {string} - The HTML content of the career path information
 */
export function getCareerPathInfoAsHtml(careerPathId) {
  debugLog(`Getting career path info for: ${careerPathId}`);
  debugLog(`Current working directory: ${process.cwd()}`);
  debugLog(`NODE_ENV: ${process.env.NODE_ENV}`);
  debugLog(`VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

  // Try multiple possible locations for the markdown file
  const possibleLocations = [
    // Standard locations
    path.join(process.cwd(), 'public', 'career_roadmaps', careerPathId, 'career_goal.md'),
    path.join(process.cwd(), 'server', 'data', 'career_roadmaps', careerPathId, 'career_goal.md'),
    path.join(process.cwd(), 'public', 'static-data', careerPathId, 'career_goal.md'),
    // Vercel-specific locations
    '/var/task/public/career_roadmaps/' + careerPathId + '/career_goal.md',
    '/var/task/server/data/career_roadmaps/' + careerPathId + '/career_goal.md',
    '/var/task/public/static-data/' + careerPathId + '/career_goal.md',
  ];

  debugLog('Checking possible locations for markdown file:');
  possibleLocations.forEach(loc => debugLog(`- ${loc}`));

  // Try each location until we find the file
  for (const markdownPath of possibleLocations) {
    try {
      if (fs.existsSync(markdownPath)) {
        debugLog(`Found markdown file at: ${markdownPath}`);
        return readMarkdownFileAsHtml(markdownPath);
      }
    } catch (err) {
      debugLog(`Error checking location ${markdownPath}:`, err.message);
    }
  }

  // If we couldn't find the file, return a fallback message
  debugLog('Could not find markdown file in any location, using fallback');

  // Generate a fallback message based on the career path ID
  const formattedName = careerPathId
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  const fallbackMarkdown = `
# ${formattedName} Career Path

## Overview
This is a placeholder for the ${formattedName} career path information.

## Requirements
- Education requirements will be listed here
- Skill requirements will be listed here
- Experience requirements will be listed here

## Career Progression
1. Entry-level positions
2. Mid-level positions
3. Senior-level positions
4. Leadership positions

## Organizations & Certifications
- Relevant organizations will be listed here
- Relevant certifications will be listed here

## Tips & Resources
- Career tips will be listed here
- Useful resources will be listed here
`;

  return convertMarkdownToHtml(fallbackMarkdown);
}

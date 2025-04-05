import fs from 'fs';
import path from 'path';
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
    const markdown = fs.readFileSync(filePath, 'utf-8');
    return convertMarkdownToHtml(markdown);
  } catch (error) {
    console.error(`Error reading markdown file at ${filePath}:`, error);
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
  const markdownPath = path.join(process.cwd(), 'public', 'career_roadmaps', careerPathId, 'career_goal.md');
  return readMarkdownFileAsHtml(markdownPath);
}

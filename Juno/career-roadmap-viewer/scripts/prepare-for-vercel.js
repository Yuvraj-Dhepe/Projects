/**
 * This script ensures that all necessary data files are included in the Vercel deployment
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get the current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Log function
function log(message) {
  console.log(`[prepare-for-vercel] ${message}`);
}

// Main function
async function main() {
  try {
    log('Starting preparation for Vercel deployment...');

    // Check if public/career_roadmaps directory exists
    const rootDir = path.resolve(__dirname, '..');
    const careerRoadmapsDir = path.resolve(rootDir, 'public/career_roadmaps');
    if (!fs.existsSync(careerRoadmapsDir)) {
      log(`Error: Directory does not exist: ${careerRoadmapsDir}`);
      process.exit(1);
    }

    // List all career paths
    const directories = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name);

    log(`Found ${directories.length} career paths: ${directories.join(', ')}`);

    // Check for diagram files
    let missingDiagrams = [];
    directories.forEach(dir => {
      const diagramPath = path.join(careerRoadmapsDir, dir, 'diagrams', 'career_path.mmd');
      if (!fs.existsSync(diagramPath)) {
        missingDiagrams.push(dir);
      }
    });

    if (missingDiagrams.length > 0) {
      log(`Warning: Missing diagram files for: ${missingDiagrams.join(', ')}`);
    } else {
      log('All career paths have diagram files.');
    }

    // Create a .vercelignore file to ensure we don't exclude anything important
    const vercelIgnorePath = path.resolve(rootDir, '.vercelignore');
    const vercelIgnoreContent = `
# Vercel ignore file
# Explicitly DO NOT ignore these directories
!public/career_roadmaps
!public/career_roadmaps/**
`;

    fs.writeFileSync(vercelIgnorePath, vercelIgnoreContent.trim());
    log('Created .vercelignore file to ensure data files are included.');

    // Create a simple file to verify the build process
    const verifyPath = path.resolve(rootDir, 'public/vercel-build-verification.txt');
    fs.writeFileSync(verifyPath, `Build verification file created at ${new Date().toISOString()}`);
    log('Created verification file.');

    log('Preparation for Vercel deployment completed successfully.');
  } catch (error) {
    log(`Error during preparation: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main();

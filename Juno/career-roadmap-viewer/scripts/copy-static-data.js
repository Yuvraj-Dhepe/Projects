/**
 * Script to copy career roadmaps data to static-data directory
 * This ensures the data is available in the Vercel deployment
 */
const fs = require('fs');
const path = require('path');

// Define source and destination directories
const sourceDir = path.resolve(__dirname, '../server/data/career_roadmaps');
const destDir = path.resolve(__dirname, '../public/static-data');

/**
 * Copy a directory recursively
 * @param {string} src - Source directory
 * @param {string} dest - Destination directory
 */
function copyDirRecursive(src, dest) {
  // Create destination directory if it doesn't exist
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
    console.log(`Created directory: ${dest}`);
  }

  // Read source directory
  const entries = fs.readdirSync(src, { withFileTypes: true });

  // Copy each entry
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    // If entry is a directory, copy recursively
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      // Copy file
      fs.copyFileSync(srcPath, destPath);
      console.log(`Copied file: ${srcPath} -> ${destPath}`);
    }
  }
}

// Main function
function main() {
  console.log('Starting to copy career roadmaps data to static-data directory...');
  console.log(`Source directory: ${sourceDir}`);
  console.log(`Destination directory: ${destDir}`);

  try {
    // Check if source directory exists
    if (!fs.existsSync(sourceDir)) {
      console.error(`Source directory does not exist: ${sourceDir}`);
      process.exit(1);
    }

    // Get list of career paths
    const careerPaths = fs.readdirSync(sourceDir, { withFileTypes: true })
      .filter(entry => entry.isDirectory())
      .map(entry => entry.name);

    console.log(`Found ${careerPaths.length} career paths: ${careerPaths.join(', ')}`);

    // Create destination directory if it doesn't exist
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
      console.log(`Created destination directory: ${destDir}`);
    }

    // Copy each career path
    for (const careerPath of careerPaths) {
      const srcCareerPath = path.join(sourceDir, careerPath);
      const destCareerPath = path.join(destDir, careerPath);

      console.log(`Copying career path: ${careerPath}`);
      copyDirRecursive(srcCareerPath, destCareerPath);
    }

    console.log('Successfully copied all career roadmaps data to static-data directory.');
  } catch (error) {
    console.error('Error copying career roadmaps data:', error);
    process.exit(1);
  }
}

// Run the main function
main();

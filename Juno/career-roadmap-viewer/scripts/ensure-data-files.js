/**
 * Script to ensure that all necessary data files are properly created and placed
 * in the correct locations for deployment.
 */

import fs from 'fs';
import path from 'path';

// Get the root directory of the project
const rootDir = process.cwd();

/**
 * Log a message to the console
 * @param {string} message - The message to log
 */
function log(message) {
  console.log(`[ensure-data-files] ${message}`);
}

/**
 * Create a directory if it doesn't exist
 * @param {string} dir - The directory to create
 */
function ensureDirectoryExists(dir) {
  if (!fs.existsSync(dir)) {
    log(`Creating directory: ${dir}`);
    fs.mkdirSync(dir, { recursive: true });
  }
}

/**
 * Copy a file from source to destination
 * @param {string} src - The source file path
 * @param {string} dest - The destination file path
 */
function copyFile(src, dest) {
  try {
    ensureDirectoryExists(path.dirname(dest));
    fs.copyFileSync(src, dest);
    log(`Copied file: ${src} -> ${dest}`);
  } catch (error) {
    log(`Error copying file ${src} to ${dest}: ${error.message}`);
  }
}

/**
 * Copy a directory recursively
 * @param {string} src - The source directory path
 * @param {string} dest - The destination directory path
 */
function copyDirRecursive(src, dest) {
  try {
    // Create destination directory if it doesn't exist
    ensureDirectoryExists(dest);

    // Get all files and directories in the source directory
    const entries = fs.readdirSync(src, { withFileTypes: true });

    // Copy each entry
    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);

      if (entry.isDirectory()) {
        // Recursively copy directory
        copyDirRecursive(srcPath, destPath);
      } else {
        // Copy file
        copyFile(srcPath, destPath);
      }
    }

    log(`Copied directory: ${src} -> ${dest}`);
  } catch (error) {
    log(`Error copying directory ${src} to ${dest}: ${error.message}`);
  }
}

/**
 * Write content to a file
 * @param {string} filePath - The file path
 * @param {string} content - The content to write
 */
function writeFile(filePath, content) {
  try {
    ensureDirectoryExists(path.dirname(filePath));
    fs.writeFileSync(filePath, content);
    log(`Wrote file: ${filePath}`);
  } catch (error) {
    log(`Error writing file ${filePath}: ${error.message}`);
  }
}

/**
 * Generate career paths data from the career_roadmaps directory
 * @return {Array} The career paths data
 */
function generateCareerPathsData() {
  try {
    const careerRoadmapsDir = path.resolve(rootDir, 'public/career_roadmaps');

    if (!fs.existsSync(careerRoadmapsDir)) {
      log(`Career roadmaps directory does not exist: ${careerRoadmapsDir}`);
      return [];
    }

    const directories = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name)
      .filter(name => !['node_modules', '.git'].includes(name));

    const careerPaths = directories
      .filter(dir => {
        const diagramPath = path.join(careerRoadmapsDir, dir, 'diagrams', 'career_path.mmd');
        return fs.existsSync(diagramPath);
      })
      .map(dir => {
        // Format the name for display (convert snake_case to Title Case)
        const name = dir
          .split('_')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');

        return {
          id: dir,
          name: name
        };
      });

    log(`Generated ${careerPaths.length} career paths from directory scan`);
    return careerPaths;
  } catch (error) {
    log(`Error generating career paths data: ${error.message}`);
    return [];
  }
}

/**
 * Main function
 */
async function main() {
  try {
    log('Starting data files check...');
    log(`Current working directory: ${rootDir}`);

    // Generate career paths data
    const careerPaths = generateCareerPathsData();
    const careerPathsJson = JSON.stringify(careerPaths, null, 2);

    // Define paths for the data files
    const publicDataDir = path.resolve(rootDir, 'public/data');
    const serverDataDir = path.resolve(rootDir, 'server/data');
    
    const publicCareerPathsFile = path.resolve(publicDataDir, 'career-paths.json');
    const serverCareerPathsFile = path.resolve(serverDataDir, 'career-paths.json');

    // Ensure directories exist
    ensureDirectoryExists(publicDataDir);
    ensureDirectoryExists(serverDataDir);

    // Write the career paths data to public and server directories
    writeFile(publicCareerPathsFile, careerPathsJson);
    writeFile(serverCareerPathsFile, careerPathsJson);

    // Copy the career_roadmaps directory to the server data directory
    const careerRoadmapsDir = path.resolve(rootDir, 'public/career_roadmaps');
    const serverCareerRoadmapsDir = path.resolve(serverDataDir, 'career_roadmaps');

    if (fs.existsSync(careerRoadmapsDir)) {
      log('Copying career_roadmaps directory to server data directory...');
      copyDirRecursive(careerRoadmapsDir, serverCareerRoadmapsDir);
      log('Successfully copied career_roadmaps to server data directory');
    } else {
      log(`Warning: Career roadmaps directory does not exist: ${careerRoadmapsDir}`);
    }

    log('Data files check completed successfully!');
  } catch (error) {
    log(`Error during data files check: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

// Run the main function
main();
/**
 * Script to ensure that all necessary data files are properly created and placed
 * in the correct locations for deployment to Vercel
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
    log(`NODE_ENV: ${process.env.NODE_ENV}`);
    log(`VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Ensure the data directory exists in all necessary locations
    const publicDataDir = path.resolve(rootDir, 'public/data');
    const outputPublicDataDir = path.resolve(rootDir, '.output/public/data');
    const serverDataDir = path.resolve(rootDir, 'server/data');

    ensureDirectoryExists(publicDataDir);
    ensureDirectoryExists(outputPublicDataDir);
    ensureDirectoryExists(serverDataDir);

    // Create Vercel-specific directories if needed
    if (process.env.VERCEL_ENV) {
      log('Running in Vercel environment, ensuring additional directories exist');
      ensureDirectoryExists('/var/task/public/career_roadmaps');
      ensureDirectoryExists('/var/task/public/static-data');
      ensureDirectoryExists('/var/task/server/data/career_roadmaps');
    }

    // Generate career paths data
    const careerPaths = generateCareerPathsData();
    const careerPathsJson = JSON.stringify(careerPaths, null, 2);

    // Write the career paths data to all necessary locations
    const publicCareerPathsFile = path.resolve(publicDataDir, 'career-paths.json');
    const outputCareerPathsFile = path.resolve(outputPublicDataDir, 'career-paths.json');
    const serverCareerPathsFile = path.resolve(serverDataDir, 'career-paths.json');

    writeFile(publicCareerPathsFile, careerPathsJson);
    writeFile(outputCareerPathsFile, careerPathsJson);
    writeFile(serverCareerPathsFile, careerPathsJson);

    // Copy the career_roadmaps directory to all necessary locations
    const careerRoadmapsDir = path.resolve(rootDir, 'public/career_roadmaps');
    const outputCareerRoadmapsDir = path.resolve(rootDir, '.output/public/career_roadmaps');
    const serverCareerRoadmapsDir = path.resolve(rootDir, 'server/data/career_roadmaps');
    const staticDataDir = path.resolve(rootDir, 'public/static-data');
    const outputStaticDataDir = path.resolve(rootDir, '.output/public/static-data');

    if (fs.existsSync(careerRoadmapsDir)) {
      log('Copying career_roadmaps directory to all necessary locations...');

      // Copy to output directory
      log(`Copying to output directory: ${outputCareerRoadmapsDir}`);
      copyDirRecursive(careerRoadmapsDir, outputCareerRoadmapsDir);

      // Copy to server data directory
      log(`Copying to server data directory: ${serverCareerRoadmapsDir}`);
      copyDirRecursive(careerRoadmapsDir, serverCareerRoadmapsDir);

      // Copy to static-data directory (for Vercel deployment)
      log(`Copying to static-data directory: ${staticDataDir}`);
      copyDirRecursive(careerRoadmapsDir, staticDataDir);

      // Copy to output static-data directory
      log(`Copying to output static-data directory: ${outputStaticDataDir}`);
      copyDirRecursive(careerRoadmapsDir, outputStaticDataDir);

      log('Successfully copied career_roadmaps to all locations');
    } else {
      log(`Warning: Career roadmaps directory does not exist: ${careerRoadmapsDir}`);
    }

    // Create a verification file in each location to confirm they exist
    const verificationContent = `Data files check completed at ${new Date().toISOString()}\nFound ${careerPaths.length} career paths: ${careerPaths.map(p => p.id).join(', ')}`;

    // Create verification files in each career_roadmaps directory
    writeFile(path.resolve(careerRoadmapsDir, 'verification.txt'), verificationContent);
    writeFile(path.resolve(outputCareerRoadmapsDir, 'verification.txt'), verificationContent);
    writeFile(path.resolve(serverCareerRoadmapsDir, 'verification.txt'), verificationContent);
    writeFile(path.resolve(staticDataDir, 'verification.txt'), verificationContent);
    writeFile(path.resolve(outputStaticDataDir, 'verification.txt'), verificationContent);

    // Create verification files in data directories
    writeFile(path.resolve(publicDataDir, 'verification.txt'), verificationContent);
    writeFile(path.resolve(outputPublicDataDir, 'verification.txt'), verificationContent);
    writeFile(path.resolve(serverDataDir, 'verification.txt'), verificationContent);

    // Create a special verification file for Vercel
    const vercelVerificationContent = `Vercel build verification file created at ${new Date().toISOString()}\nThis file confirms that the build process ran successfully.`;
    writeFile(path.resolve(rootDir, 'public/vercel-build-verification.txt'), vercelVerificationContent);

    log('Data files check completed successfully!');
  } catch (error) {
    log(`Error during data files check: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

// Run the main function
main();

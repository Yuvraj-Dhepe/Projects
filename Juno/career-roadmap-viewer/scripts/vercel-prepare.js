/**
 * Special script to prepare the Vercel environment
 * This script runs during the Vercel build process
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get the root directory of the project
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

/**
 * Log a message to the console
 * @param {string} message - The message to log
 */
function log(message) {
  console.log(`[vercel-prepare] ${message}`);
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
 * Main function
 */
async function main() {
  try {
    log('Starting Vercel preparation...');
    log(`Current working directory: ${rootDir}`);
    log(`NODE_ENV: ${process.env.NODE_ENV}`);
    log(`VERCEL_ENV: ${process.env.VERCEL_ENV || 'not set'}`);

    // Create special directories for Vercel
    const vercelDirs = [
      path.resolve(rootDir, '.vercel/output/static'),
      path.resolve(rootDir, '.vercel/output/static/career_roadmaps'),
      path.resolve(rootDir, '.vercel/output/static/static-data'),
      path.resolve(rootDir, '.vercel/output/static/data')
    ];

    // Create each directory
    for (const dir of vercelDirs) {
      ensureDirectoryExists(dir);
    }

    // Copy career_roadmaps to Vercel output
    const careerRoadmapsDir = path.resolve(rootDir, 'public/career_roadmaps');
    const vercelCareerRoadmapsDir = path.resolve(rootDir, '.vercel/output/static/career_roadmaps');

    if (fs.existsSync(careerRoadmapsDir)) {
      log('Copying career_roadmaps to Vercel output directory');
      copyDirRecursive(careerRoadmapsDir, vercelCareerRoadmapsDir);
    } else {
      log('Warning: career_roadmaps directory not found');
    }

    // Copy static-data to Vercel output
    const staticDataDir = path.resolve(rootDir, 'public/static-data');
    const vercelStaticDataDir = path.resolve(rootDir, '.vercel/output/static/static-data');

    if (fs.existsSync(staticDataDir)) {
      log('Copying static-data to Vercel output directory');
      copyDirRecursive(staticDataDir, vercelStaticDataDir);
    } else {
      log('Warning: static-data directory not found');

      // If static-data doesn't exist but career_roadmaps does, copy it
      if (fs.existsSync(careerRoadmapsDir)) {
        log('Copying career_roadmaps to static-data in Vercel output');
        copyDirRecursive(careerRoadmapsDir, vercelStaticDataDir);
      }
    }

    // Copy data directory to Vercel output
    const dataDir = path.resolve(rootDir, 'public/data');
    const vercelDataDir = path.resolve(rootDir, '.vercel/output/static/data');

    if (fs.existsSync(dataDir)) {
      log('Copying data directory to Vercel output');
      copyDirRecursive(dataDir, vercelDataDir);
    } else {
      log('Warning: data directory not found');
    }

    // Create verification file
    const verificationContent = `Vercel preparation completed at ${new Date().toISOString()}`;
    writeFile(path.resolve(rootDir, '.vercel/output/static/vercel-preparation.txt'), verificationContent);

    log('Vercel preparation completed successfully!');
  } catch (error) {
    log(`Error during Vercel preparation: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

// Run the main function
main();

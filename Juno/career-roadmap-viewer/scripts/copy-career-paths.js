import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get the current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Log function
function log(message) {
  console.log(`[copy-career-paths] ${message}`);
}

// Copy directory recursively
function copyDirRecursive(src, dest) {
  // Create destination directory if it doesn't exist
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }
  
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
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

// Main function
async function main() {
  try {
    log('Starting career paths copy process...');
    
    // Source directory (original career roadmaps)
    const sourceDir = path.resolve(rootDir, 'public/career_roadmaps');
    
    if (!fs.existsSync(sourceDir)) {
      log(`Error: Source directory does not exist: ${sourceDir}`);
      process.exit(1);
    }
    
    // Destination directory (public/data)
    const destDir = path.resolve(rootDir, 'public/data/career_roadmaps');
    
    if (!fs.existsSync(path.dirname(destDir))) {
      fs.mkdirSync(path.dirname(destDir), { recursive: true });
    }
    
    // Copy the entire career_roadmaps directory
    log(`Copying career paths from ${sourceDir} to ${destDir}`);
    copyDirRecursive(sourceDir, destDir);
    
    // Create a JSON file with the list of career paths
    const careerPaths = fs.readdirSync(sourceDir, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => {
        // Format the name for display (convert snake_case to Title Case)
        const name = dirent.name
          .split('_')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
        
        return {
          id: dirent.name,
          name: name
        };
      });
    
    // Write the career paths to a JSON file
    const careerPathsFile = path.resolve(rootDir, 'public/data/career-paths.json');
    fs.writeFileSync(careerPathsFile, JSON.stringify(careerPaths, null, 2));
    
    log(`Wrote ${careerPaths.length} career paths to: ${careerPathsFile}`);
    
    // Create a verification file
    const verificationFile = path.resolve(rootDir, 'public/data/verification.txt');
    fs.writeFileSync(verificationFile, `Career paths copied at ${new Date().toISOString()}\nFound ${careerPaths.length} career paths: ${careerPaths.map(p => p.id).join(', ')}`);
    
    log('Career paths copy process completed successfully!');
  } catch (error) {
    log(`Error during career paths copy: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

// Run the main function
main();

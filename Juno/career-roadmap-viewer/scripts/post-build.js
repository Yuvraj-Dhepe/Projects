import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get the current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Log function
function log(message) {
  console.log(`[post-build] ${message}`);
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
    log('Starting post-build process...');
    
    // Check if the .output directory exists
    const outputDir = path.resolve(rootDir, '.output');
    if (!fs.existsSync(outputDir)) {
      log(`Output directory does not exist: ${outputDir}`);
      log('This is normal if running locally. The script will create necessary directories for Vercel.');
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Check if the .output/public directory exists
    const outputPublicDir = path.resolve(outputDir, 'public');
    if (!fs.existsSync(outputPublicDir)) {
      log(`Output public directory does not exist: ${outputPublicDir}`);
      fs.mkdirSync(outputPublicDir, { recursive: true });
    }
    
    // Copy static data from public/static-data to .output/public/static-data
    const srcStaticDataDir = path.resolve(rootDir, 'public/static-data');
    const destStaticDataDir = path.resolve(outputPublicDir, 'static-data');
    
    if (fs.existsSync(srcStaticDataDir)) {
      log(`Copying static data from ${srcStaticDataDir} to ${destStaticDataDir}`);
      copyDirRecursive(srcStaticDataDir, destStaticDataDir);
      log('Static data copied successfully');
    } else {
      log(`Source static data directory does not exist: ${srcStaticDataDir}`);
      
      // Try to copy from public/_nuxt/static-data instead
      const nuxtStaticDataDir = path.resolve(rootDir, 'public/_nuxt/static-data');
      if (fs.existsSync(nuxtStaticDataDir)) {
        log(`Copying static data from ${nuxtStaticDataDir} to ${destStaticDataDir}`);
        copyDirRecursive(nuxtStaticDataDir, destStaticDataDir);
        log('Static data copied successfully from Nuxt directory');
      } else {
        log(`Nuxt static data directory does not exist: ${nuxtStaticDataDir}`);
        
        // Create a minimal static data file with a message
        log('Creating minimal static data file');
        fs.mkdirSync(destStaticDataDir, { recursive: true });
        fs.writeFileSync(
          path.join(destStaticDataDir, 'verification.txt'),
          `Post-build script ran at ${new Date().toISOString()}\nNo source static data found.`
        );
        
        // Create an empty career paths file
        fs.writeFileSync(
          path.join(destStaticDataDir, 'career-paths.json'),
          JSON.stringify([
            { id: 'sample_career', name: 'Sample Career' }
          ], null, 2)
        );
      }
    }
    
    // Create a verification file
    fs.writeFileSync(
      path.join(outputPublicDir, 'post-build-verification.txt'),
      `Post-build script completed at ${new Date().toISOString()}`
    );
    
    log('Post-build process completed successfully');
  } catch (error) {
    log(`Error during post-build: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

// Run the main function
main();

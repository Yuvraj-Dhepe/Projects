import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get the current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Log function
function log(message) {
  console.log(`[generate-static-data] ${message}`);
}

// Main function
async function main() {
  try {
    log('Starting generation of static data files...');

    // Path to career_roadmaps directory
    const careerRoadmapsDir = path.resolve(rootDir, 'public/career_roadmaps');

    if (!fs.existsSync(careerRoadmapsDir)) {
      log(`Error: Directory does not exist: ${careerRoadmapsDir}`);
      process.exit(1);
    }

    // Read all directories in the career_roadmaps folder
    const allItems = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true });

    const directories = allItems
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name)
      .filter(name => !['node_modules', '.git'].includes(name));

    log(`Found ${directories.length} potential career paths: ${directories.join(', ')}`);

    // Filter to only include directories that have diagrams/career_path.mmd
    const careerPaths = directories
      .filter(dir => {
        const diagramPath = path.join(careerRoadmapsDir, dir, 'diagrams', 'career_path.mmd');
        const exists = fs.existsSync(diagramPath);
        return exists;
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

    log(`Found ${careerPaths.length} valid career paths with diagrams`);

    // Create the static data directories
    // 1. In public/static-data (for local development)
    const publicStaticDataDir = path.resolve(rootDir, 'public/static-data');
    if (!fs.existsSync(publicStaticDataDir)) {
      fs.mkdirSync(publicStaticDataDir, { recursive: true });
      log(`Created public static data directory: ${publicStaticDataDir}`);
    }

    // 2. In public/_nuxt/static-data (for Vercel deployment)
    const nuxtStaticDataDir = path.resolve(rootDir, 'public/_nuxt/static-data');
    if (!fs.existsSync(nuxtStaticDataDir)) {
      fs.mkdirSync(nuxtStaticDataDir, { recursive: true });
      log(`Created Nuxt static data directory: ${nuxtStaticDataDir}`);
    }

    // 3. In .output/public/static-data (for Vercel deployment)
    const outputStaticDataDir = path.resolve(rootDir, '.output/public/static-data');
    if (!fs.existsSync(outputStaticDataDir)) {
      fs.mkdirSync(outputStaticDataDir, { recursive: true });
      log(`Created output static data directory: ${outputStaticDataDir}`);
    }

    // Write the career paths to JSON files in all locations
    const publicCareerPathsFile = path.resolve(publicStaticDataDir, 'career-paths.json');
    const nuxtCareerPathsFile = path.resolve(nuxtStaticDataDir, 'career-paths.json');
    const outputCareerPathsFile = path.resolve(outputStaticDataDir, 'career-paths.json');

    const careerPathsJson = JSON.stringify(careerPaths, null, 2);
    fs.writeFileSync(publicCareerPathsFile, careerPathsJson);
    fs.writeFileSync(nuxtCareerPathsFile, careerPathsJson);

    // The .output directory might not exist yet during build, so handle errors gracefully
    try {
      fs.writeFileSync(outputCareerPathsFile, careerPathsJson);
      log(`Wrote career paths to: ${outputCareerPathsFile}`);
    } catch (err) {
      log(`Note: Could not write to ${outputCareerPathsFile} yet: ${err.message}`);
    }

    log(`Wrote career paths to: ${publicCareerPathsFile}`);
    log(`Wrote career paths to: ${nuxtCareerPathsFile}`);

    // Copy diagram files to both static data directories
    for (const careerPath of careerPaths) {
      const sourceDir = path.join(careerRoadmapsDir, careerPath.id, 'diagrams');

      // Create target directories
      const publicTargetDir = path.join(publicStaticDataDir, careerPath.id, 'diagrams');
      const nuxtTargetDir = path.join(nuxtStaticDataDir, careerPath.id, 'diagrams');
      const outputTargetDir = path.join(outputStaticDataDir, careerPath.id, 'diagrams');

      if (!fs.existsSync(publicTargetDir)) {
        fs.mkdirSync(publicTargetDir, { recursive: true });
      }

      if (!fs.existsSync(nuxtTargetDir)) {
        fs.mkdirSync(nuxtTargetDir, { recursive: true });
      }

      try {
        if (!fs.existsSync(outputTargetDir)) {
          fs.mkdirSync(outputTargetDir, { recursive: true });
        }
      } catch (err) {
        log(`Note: Could not create ${outputTargetDir} yet: ${err.message}`);
      }

      // Copy the diagram files
      const diagramFiles = ['career_path.mmd', 'career_path.png', 'career_path.svg'].filter(file =>
        fs.existsSync(path.join(sourceDir, file))
      );

      for (const file of diagramFiles) {
        const sourceFile = path.join(sourceDir, file);
        const publicTargetFile = path.join(publicTargetDir, file);
        const nuxtTargetFile = path.join(nuxtTargetDir, file);
        const outputTargetFile = path.join(outputTargetDir, file);

        fs.copyFileSync(sourceFile, publicTargetFile);
        fs.copyFileSync(sourceFile, nuxtTargetFile);

        try {
          fs.copyFileSync(sourceFile, outputTargetFile);
          log(`Copied ${file} for ${careerPath.id} to output location`);
        } catch (err) {
          log(`Note: Could not copy to ${outputTargetFile} yet: ${err.message}`);
        }

        log(`Copied ${file} for ${careerPath.id} to public and nuxt locations`);
      }

      // Copy the career goal markdown file if it exists
      const goalFile = path.join(careerRoadmapsDir, careerPath.id, 'career_goal.md');
      if (fs.existsSync(goalFile)) {
        const publicGoalFile = path.join(publicStaticDataDir, careerPath.id, 'career_goal.md');
        const nuxtGoalFile = path.join(nuxtStaticDataDir, careerPath.id, 'career_goal.md');
        const outputGoalFile = path.join(outputStaticDataDir, careerPath.id, 'career_goal.md');

        fs.copyFileSync(goalFile, publicGoalFile);
        fs.copyFileSync(goalFile, nuxtGoalFile);

        try {
          fs.copyFileSync(goalFile, outputGoalFile);
          log(`Copied career_goal.md for ${careerPath.id} to output location`);
        } catch (err) {
          log(`Note: Could not copy to ${outputGoalFile} yet: ${err.message}`);
        }

        log(`Copied career_goal.md for ${careerPath.id} to public and nuxt locations`);
      }
    }

    // Create a verification file
    const verificationContent = `Static data generated at ${new Date().toISOString()}
Found ${careerPaths.length} career paths: ${careerPaths.map(p => p.id).join(', ')}`;

    fs.writeFileSync(path.join(publicStaticDataDir, 'verification.txt'), verificationContent);
    fs.writeFileSync(path.join(nuxtStaticDataDir, 'verification.txt'), verificationContent);

    try {
      fs.writeFileSync(path.join(outputStaticDataDir, 'verification.txt'), verificationContent);
    } catch (err) {
      log(`Note: Could not write verification file to output location: ${err.message}`);
    }

    log('Static data generation completed successfully!');
  } catch (error) {
    log(`Error during static data generation: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}

// Run the main function
main();

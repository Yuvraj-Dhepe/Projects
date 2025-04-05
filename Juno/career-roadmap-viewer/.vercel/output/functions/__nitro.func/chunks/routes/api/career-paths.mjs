import { d as defineEventHandler, u as useRuntimeConfig } from '../../nitro/nitro.mjs';
import fs from 'fs';
import path from 'path';
import 'node:http';
import 'node:https';
import 'node:events';
import 'node:buffer';
import 'node:fs';
import 'node:path';
import 'node:crypto';

const config = useRuntimeConfig();
const DEBUG = config.debugMode === "true";
function debugLog(message, ...args) {
  if (DEBUG) {
    console.log(message, ...args);
  }
}
const careerPaths = defineEventHandler(async (event) => {
  try {
    const careerRoadmapsDir = path.resolve(process.cwd(), "public/career_roadmaps");
    debugLog("Looking for career paths in:", careerRoadmapsDir);
    if (!fs.existsSync(careerRoadmapsDir)) {
      console.error("Directory does not exist:", careerRoadmapsDir);
      return [];
    }
    const allItems = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true });
    debugLog("All items in directory:", allItems.map((item) => item.name));
    const directories = allItems.filter((dirent) => dirent.isDirectory()).map((dirent) => dirent.name).filter((name) => !["node_modules", ".git"].includes(name));
    debugLog("Filtered directories:", directories);
    const careerPaths = directories.filter((dir) => {
      const diagramPath = path.join(careerRoadmapsDir, dir, "diagrams", "career_path.mmd");
      const exists = fs.existsSync(diagramPath);
      debugLog(`Checking for diagram in ${dir}:`, diagramPath, exists ? "FOUND" : "NOT FOUND");
      return exists;
    }).map((dir) => {
      const name = dir.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
      return {
        id: dir,
        name
      };
    });
    debugLog("Final career paths:", careerPaths);
    debugLog("API Response:", JSON.stringify(careerPaths));
    return Array.isArray(careerPaths) ? careerPaths : [];
  } catch (error) {
    console.error("Error discovering career paths:", error);
    return { error: "Failed to discover career paths", message: error.message };
  }
});

export { careerPaths as default };
//# sourceMappingURL=career-paths.mjs.map

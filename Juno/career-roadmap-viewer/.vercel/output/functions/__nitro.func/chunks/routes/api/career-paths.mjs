import { d as defineEventHandler } from '../../nitro/nitro.mjs';
import fs from 'fs';
import path from 'path';
import 'node:http';
import 'node:https';
import 'node:events';
import 'node:buffer';
import 'node:fs';
import 'node:path';
import 'node:crypto';

const careerPaths = defineEventHandler(async (event) => {
  try {
    const careerRoadmapsDir = path.resolve(process.cwd(), "../career_roadmaps");
    const directories = fs.readdirSync(careerRoadmapsDir, { withFileTypes: true }).filter((dirent) => dirent.isDirectory()).map((dirent) => dirent.name).filter((name) => !["node_modules", ".git"].includes(name));
    const careerPaths = directories.filter((dir) => {
      const diagramPath = path.join(careerRoadmapsDir, dir, "diagrams", "career_path.mmd");
      return fs.existsSync(diagramPath);
    }).map((dir) => {
      const name = dir.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
      return {
        id: dir,
        name
      };
    });
    return careerPaths;
  } catch (error) {
    console.error("Error discovering career paths:", error);
    return { error: "Failed to discover career paths", message: error.message };
  }
});

export { careerPaths as default };
//# sourceMappingURL=career-paths.mjs.map

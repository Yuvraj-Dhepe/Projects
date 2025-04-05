import { d as defineEventHandler, g as getRouterParam, s as setResponseStatus } from '../../../nitro/nitro.mjs';
import fs from 'fs';
import path from 'path';
import 'node:http';
import 'node:https';
import 'node:events';
import 'node:buffer';
import 'node:fs';
import 'node:path';
import 'node:crypto';

const _path_ = defineEventHandler(async (event) => {
  try {
    const careerPath = getRouterParam(event, "path");
    const careerRoadmapsDir = path.resolve(process.cwd(), "public/career_roadmaps");
    const mermaidPath = path.join(careerRoadmapsDir, careerPath, "diagrams", "career_path.mmd");
    if (!fs.existsSync(mermaidPath)) {
      throw new Error(`Mermaid file not found: ${mermaidPath}`);
    }
    const mermaidCode = fs.readFileSync(mermaidPath, "utf-8");
    return { code: mermaidCode };
  } catch (error) {
    console.error("Error getting Mermaid code:", error);
    setResponseStatus(event, 404);
    return { error: "Mermaid code not found", message: error.message };
  }
});

export { _path_ as default };
//# sourceMappingURL=_path_.mjs.map

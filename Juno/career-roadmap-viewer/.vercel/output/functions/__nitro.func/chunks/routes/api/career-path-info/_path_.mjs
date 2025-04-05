import { d as defineEventHandler, g as getRouterParam, s as setResponseStatus } from '../../../nitro/nitro.mjs';
import fs from 'fs';
import path from 'path';
import { marked } from 'marked';
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
    const careerRoadmapsDir = path.resolve(process.cwd(), "../career_roadmaps");
    const readmePath = path.join(careerRoadmapsDir, careerPath, "README.md");
    if (!fs.existsSync(readmePath)) {
      return { content: null };
    }
    const fileContent = fs.readFileSync(readmePath, "utf-8");
    const htmlContent = marked(fileContent);
    return { content: htmlContent };
  } catch (error) {
    console.error("Error getting career path info:", error);
    setResponseStatus(event, 500);
    return { error: "Failed to get career path info", message: error.message };
  }
});

export { _path_ as default };
//# sourceMappingURL=_path_.mjs.map

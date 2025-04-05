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
    const careerRoadmapsDir = path.resolve(process.cwd(), "public/career_roadmaps");
    const careerGoalPath = path.join(careerRoadmapsDir, careerPath, "career_goal.md");
    const readmePath = path.join(careerRoadmapsDir, careerPath, "README.md");
    let filePath;
    if (fs.existsSync(careerGoalPath)) {
      filePath = careerGoalPath;
    } else if (fs.existsSync(readmePath)) {
      filePath = readmePath;
    } else {
      return { content: null };
    }
    const fileContent = fs.readFileSync(filePath, "utf-8");
    const htmlContent = marked(fileContent);
    const styledHtmlContent = `
      <div class="career-info">
        ${htmlContent}
      </div>
    `;
    return { content: styledHtmlContent };
  } catch (error) {
    console.error("Error getting career path info:", error);
    setResponseStatus(event, 500);
    return { error: "Failed to get career path info", message: error.message };
  }
});

export { _path_ as default };
//# sourceMappingURL=_path_.mjs.map

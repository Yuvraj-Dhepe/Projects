import { d as defineEventHandler, g as getRouterParam, a as getQuery, b as setResponseHeader, s as setResponseStatus } from '../../../nitro/nitro.mjs';
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
    const query = getQuery(event);
    const format = query.format || "png";
    if (!["png", "svg"].includes(format)) {
      throw new Error("Invalid format. Must be png or svg.");
    }
    const careerRoadmapsDir = path.resolve(process.cwd(), "../career_roadmaps");
    const diagramPath = path.join(careerRoadmapsDir, careerPath, "diagrams", `career_path.${format}`);
    if (!fs.existsSync(diagramPath)) {
      throw new Error(`Diagram file not found: ${diagramPath}`);
    }
    const fileContent = fs.readFileSync(diagramPath);
    const contentType = format === "png" ? "image/png" : "image/svg+xml";
    setResponseHeader(event, "Content-Type", contentType);
    return fileContent;
  } catch (error) {
    console.error("Error serving diagram:", error);
    setResponseStatus(event, 404);
    return { error: "Diagram not found", message: error.message };
  }
});

export { _path_ as default };
//# sourceMappingURL=_path_.mjs.map

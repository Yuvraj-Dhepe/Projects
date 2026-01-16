import re
import tomllib

# --- Load lockfile ---
with open("uv.lock", "rb") as f:
    lock = tomllib.load(f)

# Build dict: name → version
locked_versions = {}
for pkg in lock.get("package", []):
    name = pkg["name"].lower().replace("_", "-")
    version = pkg["version"]
    locked_versions[name] = version

# --- Update pyproject.toml lines ---
with open("pyproject.toml", encoding="utf-8") as f:
    lines = f.readlines()

updated_lines = []
dep_pattern = re.compile(r'^\s*"([a-zA-Z0-9\-_]+)(.*)",?$')

for line in lines:
    match = dep_pattern.match(line.strip())
    if match:
        name = match.group(1).lower().replace("_", "-")
        if name in locked_versions:
            # Replace with exact pinned version
            version = locked_versions[name]
            line = re.sub(r"(==|>=|>|<|<=)[\w\.\-]+", f"=={version}", line)
            if "==" not in line:
                # handle loose specs like "^1.2" or no version
                line = re.sub(r'("$)', f'=={version}"', line)
    updated_lines.append(line)

# --- Write updated file ---
with open("pyproject.toml", "w", encoding="utf-8") as f:
    f.writelines(updated_lines)

print("✅ Updated pyproject.toml with versions from uv.lock")

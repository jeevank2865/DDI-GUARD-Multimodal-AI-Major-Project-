#!/usr/bin/env python3
"""Extract and syntax-check the inline console JavaScript with Node.js."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


source = Path(__file__).resolve().parents[1] / "ddi_guard/static/index.html"
html = source.read_text(encoding="utf-8")
scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
if len(scripts) != 1:
    raise SystemExit(f"expected exactly one inline script in {source}, found {len(scripts)}")
node = shutil.which("node")
if node is None:
    raise SystemExit("Node.js is required for this frontend syntax check")
with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
    handle.write(scripts[0])
    handle.flush()
    subprocess.run([node, "--check", handle.name], check=True)
print(f"JavaScript syntax OK: {source}")

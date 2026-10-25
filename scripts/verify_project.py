"""Run the agreed research/demo QA checks and record their actual outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COMMANDS = [
    [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"],
    [PYTHON, "-m", "compileall", "-q", "ddi_guard", "tests", "scripts"],
    [PYTHON, "scripts/check_frontend_syntax.py"],
    [PYTHON, "scripts/paired_multimodal_validation.py"],
]


def main() -> int:
    results = []
    test_count = None
    for command in COMMANDS:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = (proc.stdout + "\n" + proc.stderr).strip()
        if "unittest" in command:
            match = re.search(r"Ran (\d+) tests?", output)
            test_count = int(match.group(1)) if match else None
        results.append({"command": " ".join(command), "exit_code": proc.returncode,
                        "status": "passed" if proc.returncode == 0 else "failed",
                        "output_tail": output[-1200:]})
        print(f"{results[-1]['status'].upper()}: {results[-1]['command']} (exit {proc.returncode})")
        if proc.returncode:
            print(results[-1]["output_tail"])
    status = "passed" if all(item["exit_code"] == 0 for item in results) and test_count is not None else "failed"
    report = {"status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "python_executable": PYTHON, "test_count": test_count, "checks": results}
    report_path = ROOT / "artifacts/qa_status.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"QA status saved: {report_path}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

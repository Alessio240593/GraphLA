import subprocess
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
scripts = [
    "network_analysis.py",
    "clustering.py",
    "training_set_creation.py",
    "models_training.py"
]

for script in scripts:
    print(f"Executing {script}...")
    result = subprocess.run(
        ["python", str(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT_DIR)}
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error {script}:")
        print(result.stderr)
        break

"""Offline validation against the installed official Databricks CLI's bundle schema."""

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import yaml

root = Path(__file__).resolve().parents[1]
cli = sys.argv[1] if len(sys.argv) > 1 else "databricks"
schema = json.loads(subprocess.check_output([cli, "bundle", "schema"], text=True))
configuration = yaml.safe_load((root / "databricks.yml").read_text())
configuration.update(yaml.safe_load((root / "resources/jobs.yml").read_text()))
jsonschema.Draft202012Validator(schema).validate(configuration)
print("Bundle matches the official CLI schema; authenticated workspace validation is separate.")

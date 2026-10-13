import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_job_wires_task_artifacts_and_workspace():
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())
    job = yaml.safe_load((ROOT / "resources/jobs.yml").read_text())["resources"]["jobs"][
        "retail_forecasting"
    ]
    assert bundle["targets"]["dev"]["workspace"]["host"] == (
        "https://dbc-9b6ad1bb-15c7.cloud.databricks.com"
    )
    assert "schedule" not in job and "job_clusters" not in job
    tasks = {task["task_key"]: task for task in job["tasks"]}
    for name, upstream in [("analytics", "ingest"), ("train", "analytics"), ("forecast", "train")]:
        assert tasks[name]["depends_on"] == [{"task_key": upstream}]
    forecast_parameters = tasks["forecast"]["notebook_task"]["base_parameters"]
    assert forecast_parameters["model_path"] == "{{tasks.train.values.model_path}}"
    for task in tasks.values():
        assert "libraries" not in task
        notebook = (ROOT / "resources" / task["notebook_task"]["notebook_path"]).resolve()
        text = notebook.read_text()
        assert text.startswith("# Databricks notebook source")
        assert '# MAGIC %pip install -r "$requirements_path"' in text
        assert text.index("restartPython()") < text.index("from retail_forecast")
        assert task["notebook_task"]["base_parameters"]["project_path"] == "${workspace.file_path}"


def test_adf_waits_for_completion_and_uses_managed_identity():
    pipeline = json.loads((ROOT / "infra/azure/adf_pipeline.json").read_text())["properties"]
    tasks = {task["name"]: task for task in pipeline["activities"]}
    start = tasks["RunDatabricksJob"]["typeProperties"]
    assert start["authentication"]["type"] == "MSI"
    assert "idempotency_token" in start["body"]["value"]
    loop = tasks["PollDatabricksRun"]["typeProperties"]
    assert all(
        state in loop["expression"]["value"]
        for state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]
    )
    poll = next(task for task in loop["activities"] if task["name"] == "GetRun")
    assert poll["typeProperties"]["authentication"] == start["authentication"]
    gate = tasks["RequireSuccessfulRun"]["typeProperties"]
    assert "SUCCESS" in gate["expression"]["value"]
    assert gate["ifFalseActivities"][0]["type"] == "Fail"

"""
Kedro project generator.

Walks project.steps, renders Jinja2 templates from templates/kedro/, writes a
complete Kedro project layout to a temp directory, then zips it for download.

TODO:
  1. Read templates from templates/kedro/ via pathlib
  2. Build a template context from project (steps, target, task_type, metrics)
  3. Render each .j2 template with jinja2.Environment
  4. Write rendered files into a temp output_dir with proper kedro layout
  5. Zip the directory with shutil.make_archive
  6. Return the zip path
"""
from __future__ import annotations

from pathlib import Path


def generate_kedro_project(project: object, output_dir: Path) -> Path:
    raise NotImplementedError(
        "Kedro export not yet implemented — see module docstring for TODO list."
    )


"""Auto-generated Kedro pipeline from ML Studio.

Each preprocessing / feature-engineering step is wired as a Kedro node with
its parameters baked in via functools.partial. To change a parameter, edit
the partial call here and re-run `kedro run`.
"""
from __future__ import annotations

from functools import partial

from kedro.pipeline import Pipeline, node, pipeline


from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            
            node(
                func=train_model,
                inputs="raw_data",
                outputs="model",
                name="train_model",
            ),
        ]
    )

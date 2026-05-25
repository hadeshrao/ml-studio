"""
LLM client for AI-powered ML interpretation.

Builds a rich system prompt from project state (dataset stats, pipeline steps,
model metrics, feature importances) and streams Claude's response token by token.

The page calls interpret() and renders with st.write_stream(). The conversation
history is maintained by the page in session state and passed in on each call,
giving Claude full multi-turn context.

No Streamlit imports — this module must be importable outside the app.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from src.state import Project

_SYSTEM_PREAMBLE = """\
You are an expert ML practitioner embedded in ML Studio, a tool that guides \
data scientists through end-to-end machine learning workflows.

You have full visibility into the user's current project — dataset, preprocessing \
pipeline, feature engineering steps, trained model, and performance metrics. \
This context is provided below and reflects the live state of their session.

Guidelines:
- Reference actual column names, step names, metric values, and hyperparameters \
from the context — never give generic advice that ignores what's in front of you.
- Use markdown: headers, bullet points, bold for emphasis, inline code for column \
and function names.
- Be specific and actionable. If something looks wrong or suboptimal, say so clearly.
- If the user asks about something not covered by the context (e.g. a model that \
hasn't been trained), note the gap and suggest how to fill it.
- Keep responses focused. Go deeper only when explicitly asked.
"""


def _serialize_project(project: "Project", domain_context: str) -> str:
    """Return a markdown-formatted snapshot of the project for the system prompt."""
    import pandas as pd

    lines: list[str] = ["## Current Project State\n"]

    # ── Dataset ───────────────────────────────────────────────────────────────
    lines.append("### Dataset")
    if project.df is not None:
        df = project.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        lines.append(f"- Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        lines.append(f"- Numeric columns: {len(num_cols)},  Categorical: {len(cat_cols)}")
        if project.target:
            lines.append(
                f"- Target: **`{project.target}`**  "
                f"(task: {project.task_type},  problem: {project.problem_type})"
            )

        # Missing values
        miss = df.isna().sum()
        miss_nonzero = miss[miss > 0].sort_values(ascending=False)
        if miss_nonzero.empty:
            lines.append("- Missing values: **none**")
        else:
            lines.append(
                f"- Missing values: {len(miss_nonzero)} columns affected, "
                f"{miss_nonzero.sum():,} total cells"
            )
            for col, cnt in miss_nonzero.head(6).items():
                lines.append(f"  - `{col}`: {cnt:,} ({cnt / len(df) * 100:.1f}%)")

        # Numeric statistics (cap at 15 columns)
        stat_cols = num_cols[:15]
        if stat_cols:
            desc = df[stat_cols].describe().T[["mean", "std", "min", "max"]]
            lines.append("\n**Numeric column statistics (up to 15 columns)**")
            lines.append("| Column | Mean | Std | Min | Max |")
            lines.append("|--------|-----:|----:|----:|----:|")
            for col, row in desc.iterrows():
                def _fmt(x: float) -> str:
                    return f"{x:.4g}" if abs(x) < 1e6 else f"{x:.2e}"
                lines.append(
                    f"| `{col}` | {_fmt(row['mean'])} | {_fmt(row['std'])} "
                    f"| {_fmt(row['min'])} | {_fmt(row['max'])} |"
                )

        # Categorical top values (cap at 5 columns × 3 values)
        if cat_cols:
            lines.append("\n**Categorical columns (top values)**")
            for col in cat_cols[:5]:
                vc = df[col].value_counts().head(3)
                top = ", ".join(f"`{v}` ({c:,})" for v, c in vc.items())
                lines.append(f"- `{col}` ({df[col].nunique()} unique): {top}")
    else:
        lines.append("- *No dataset loaded.*")

    # ── Domain context ────────────────────────────────────────────────────────
    lines.append(
        f"\n### Domain Context\n"
        f"{domain_context.strip() if domain_context.strip() else '*Not provided.*'}"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    transform_steps = [s for s in project.steps if s["stage"] != "modeling"]
    model_step = next((s for s in project.steps if s["stage"] == "modeling"), None)

    if transform_steps:
        lines.append(f"\n### Preprocessing & Feature Engineering ({len(transform_steps)} steps)")
        for i, step in enumerate(transform_steps, 1):
            params_clean = {
                k: v for k, v in step["params"].items() if not k.startswith("_")
            }
            lines.append(f"{i}. **[{step['stage']}]** `{step['name']}` — {step['description']}")
            if params_clean:
                lines.append(f"   params: `{params_clean}`")
    else:
        lines.append("\n### Pipeline\n*No preprocessing or feature engineering steps recorded.*")

    # ── Model ─────────────────────────────────────────────────────────────────
    if project.model is not None:
        algo_label = project.metrics.get("_algo_label", project.metrics.get("algorithm", "Unknown"))
        lines.append(f"\n### Trained Model\n- Algorithm: **{algo_label}**")

        # Hyperparams from the modeling step
        if model_step:
            hp = {
                k: v
                for k, v in model_step["params"].items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
            }
            if hp:
                lines.append(f"- Hyperparameters: `{hp}`")

        # Metrics
        metric_display = [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "r2", "rmse", "mae", "mape",
            "silhouette", "n_clusters", "inertia",
            "cv_mean", "cv_std", "cv_scoring", "n_train", "n_test",
        ]
        for k in metric_display:
            v = project.metrics.get(k)
            if v is None:
                continue
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"- {k}: **{fmt}**")

        # Features
        if project.features:
            feat_preview = ", ".join(f"`{f}`" for f in project.features[:20])
            suffix = f"… and {len(project.features) - 20} more" if len(project.features) > 20 else ""
            lines.append(f"\nFeature columns ({len(project.features)}): {feat_preview}{suffix}")

        # Feature importances
        try:
            from src.modeling import compute_feature_importance
            fi = compute_feature_importance(project.model, project.features or [])
            if fi is not None and len(fi) > 0:
                lines.append("\n**Top feature importances**")
                lines.append("| Feature | Importance |")
                lines.append("|---------|----------:|")
                for _, row in fi.head(10).iterrows():
                    lines.append(f"| `{row['feature']}` | {row['importance']:.4f} |")
        except Exception:
            pass
    else:
        lines.append("\n### Model\n*No model trained yet.*")

    return "\n".join(lines)


def interpret(
    project: "Project",
    api_key: str,
    domain_context: str,
    messages: list[dict],
) -> Iterator[str]:
    """
    Stream tokens from Claude given the full conversation history.

    `messages` is the complete list of {"role": "user"/"assistant", "content": "..."}
    dicts, with the new user message already appended as the last item.
    """
    import anthropic

    project_context = _serialize_project(project, domain_context)
    system_prompt = f"{_SYSTEM_PREAMBLE}\n\n{project_context}"

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

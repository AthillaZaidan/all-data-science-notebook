"""Merge each domain's EDA and pipeline notebooks into one end-to-end notebook.

Run after the domain builders:  python build_full_notebooks.py
The EDA part writes eda-output/eda_decisions.json, and the pipeline part reads it through EDA_DIR,
so every 'auto' decision in the pipeline still comes from the EDA findings above it.
"""
import copy
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
HR = "---"

FULL = [
    # (output, EDA notebook, pipeline notebook, title, description)
    ("tabular/full_tabular.ipynb", "tabular/eda_tabular.ipynb", "tabular/pipeline_tabular.ipynb",
     "Tabular End-to-End Notebook (EDA + Pipeline)",
     "Exploratory analysis and the full modelling pipeline in one run, for any tabular regression or classification task."),
    ("nlp/full_nlp.ipynb", "nlp/eda_nlp.ipynb", "nlp/pipeline_nlp.ipynb",
     "NLP End-to-End Notebook (EDA + Pipeline)",
     "Text exploratory analysis and the TF-IDF plus transformer pipeline in one run, for any text classification or regression task."),
    ("forecasting/full_forecasting.ipynb", "forecasting/eda_forecasting.ipynb", "forecasting/pipeline_forecasting.ipynb",
     "Forecasting End-to-End Notebook (EDA + Pipeline)",
     "Time-series exploratory analysis and the backtested forecasting pipeline in one run, for one or many series."),
    ("multimodal/tabular-text/full_multimodal.ipynb", "multimodal/tabular-text/eda_multimodal.ipynb", "multimodal/tabular-text/pipeline_multimodal.ipynb",
     "Multimodal End-to-End Notebook (EDA + Pipeline)",
     "Tabular + text exploratory analysis, the modality signal check, and the multimodal pipeline in one run."),
    ("multimodal/forecast-text/full_forecast_text.ipynb", "multimodal/forecast-text/eda_forecast_text.ipynb",
     "multimodal/forecast-text/pipeline_forecast_text.ipynb",
     "Forecasting + Text End-to-End Notebook (EDA + Pipeline)",
     "Time-series and text exploratory analysis, the text lead analysis, and the forecasting pipeline with text features in one run."),
    ("tabular-foundation/full_foundation.ipynb", "tabular/eda_tabular.ipynb", "tabular-foundation/pipeline_foundation.ipynb",
     "Tabular Foundation Model End-to-End Notebook (EDA + Causilo Pipeline)",
     "Tabular exploratory analysis followed by the Causilo foundation model pipeline in one run."),
]

SECTION = re.compile(r'^---\n\n# (.+?) <a name="(\d+)"></a>\n\n---$')

INHERIT = (
    "\n"
    "# reuse the data files resolved in Part 1, so the paths are edited in one place only\n"
    "CFG.DATA_PATH = str(DATA_PATH)\n"
    "CFG.TEST_PATH = str(TEST_PATH) if globals().get('TEST_PATH') else None\n"
    "print(f'Pipeline data inherited from Part 1: {CFG.DATA_PATH} | test: {CFG.TEST_PATH}')"
)


def src(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def md(text):
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": text}


def sections(cells):
    out = []
    for i, c in enumerate(cells):
        m = SECTION.match(src(c)) if c["cell_type"] == "markdown" else None
        if m:
            out.append((i, m.group(1), int(m.group(2))))
    return out


def build(out_rel, eda_rel, pipe_rel, title, desc):
    eda = json.loads((ROOT / eda_rel).read_text(encoding="utf-8"))
    pipe = json.loads((ROOT / pipe_rel).read_text(encoding="utf-8"))
    e_cells, p_cells = eda["cells"], copy.deepcopy(pipe["cells"])

    e_secs, p_secs = sections(e_cells), sections(p_cells)
    n_eda = len(e_secs)

    # pipeline part: drop its banner (3 cells) and its Introduction, renumber the remaining sections
    p_start = p_secs[1][0]
    configure = [c for c in p_cells[p_secs[0][0]:p_start] if src(c).startswith(("## How to Configure", "## Metric"))]
    body = p_cells[p_start:]
    renumber = {}
    for _, name, num in p_secs[1:]:
        renumber[num] = n_eda + num - 1
    for c in body:
        m = SECTION.match(src(c)) if c["cell_type"] == "markdown" else None
        if m:
            name, num = m.group(1), int(m.group(2))
            label = "Pipeline Initialization" if name == "Initialization" else name
            c["source"] = f'{HR}\n\n# {label} <a name="{renumber[num]}"></a>\n\n{HR}'
        if c["cell_type"] == "code" and "class Settings" in src(c) and "CFG = Settings()" in src(c):
            c["source"] = src(c) + INHERIT
        c["id"] = uuid4().hex[:8]

    toc = [f"{HR}\n\n## Table of Contents\n\n**Part 1: Exploratory Data Analysis**\n"]
    toc += [f"{num}. [**{name}**](#{num})" for _, name, num in e_secs]
    toc += ["\n**Part 2: Modelling Pipeline** (reads `eda-output/eda_decisions.json` written by Part 1)\n"]
    toc += [f"{renumber[num]}. [**{'Pipeline Initialization' if name == 'Initialization' else name}**](#{renumber[num]})"
            for _, name, num in p_secs[1:]]

    banner = md(f"{HR}\n\n# {title}\n\n*{desc} Part 1 explores the data and writes "
                f"`eda-output/eda_decisions.json`; Part 2 reads it, so every `'auto'` setting in the pipeline follows "
                f"the EDA findings. Outputs go to `eda-output/` and `pipeline-output/`.*")
    team = copy.deepcopy(e_cells[1])
    team["id"] = uuid4().hex[:8]
    bridge = md(
        f"{HR}\n\n"
        "# Part 2: Modelling Pipeline\n\n"
        f"{HR}\n\n"
        "*Everything above is the EDA; everything below is the pipeline. The pipeline reloads its own "
        "libraries and settings, inherits the data files resolved in Part 1, and reads "
        "`eda-output/eda_decisions.json` through `EDA_DIR`, so its target, task, columns to drop, "
        "validation scheme, metric, and transforms are the ones the EDA justified. Settings that are not "
        "on `'auto'` override the EDA, and the decision table records the source of every choice.*\n\n"
        "*To use the pipeline settings, edit the `Settings` cell in the Pipeline Initialization section "
        "(models, tuning, ensembling). Data paths only need to be set once, in Part 1.*"
    )
    cells = [banner, team, md("\n".join(toc))] + e_cells[3:] + [bridge] + configure + body
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": eda["metadata"], "cells": cells}
    out = ROOT / out_rel
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(cells)} cells ({n_eda} EDA + {len(p_secs) - 1} pipeline sections) -> {out}")


if __name__ == "__main__":
    for spec in FULL:
        build(*spec)

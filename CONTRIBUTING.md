# Contributing to LLM Workout

Thanks for your interest in improving this project! It's an educational resource,
so contributions that make a concept **clearer**, **more correct**, or **easier to
run** are all hugely welcome.

## Project layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Self-contained teaching notebooks, one concept each (`NN_topic.ipynb`). |
| `docs/` | The non-code "textbook" — analogies and theory behind the notebooks. |
| `src/llm_workout/` | The reusable, production-style library distilled from the notebooks. |
| `scripts/` | Runnable end-to-end demos (`train_tiny.py`, `generate.py`). |
| `tests/` | Pytest suite that guards the library's core invariants. |

### How the pieces relate

- **Notebooks teach; the library ships.** Notebooks are intentionally standalone so a
  reader can run any one in isolation, so they re-implement concepts inline rather
  than importing `llm_workout`. The library in `src/` is the polished, tested
  distillation. When you change the architecture in one place, check whether the
  other should follow.
- **Docs chapters map to notebook modules.** `docs/0X_*.md` corresponds to the
  modules in the README roadmap. Keep the wording in sync when you add a notebook.

## Development setup

```bash
python3 -m venv venv
source venv/bin/activate

# Library + test tooling
pip install -e ".[dev]"

# Add the notebook stack if you'll work on notebooks
pip install -e ".[notebooks]"
```

## Running tests

```bash
pytest -q
```

CI runs the same suite on Python 3.9–3.12 for every pull request. Please make sure
tests pass locally first, and add a test when you fix a bug or add a feature to the
library. The KV-cache equivalence test in `tests/test_model.py` is a good template
for "the optimized path must match the simple path" checks.

## Conventions

- **Notebook naming:** `NN_topic.ipynb`, zero-padded, in teaching order.
- **Keep notebooks runnable top-to-bottom** with no hidden state. Clear outputs
  before committing (`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`).
- **Style:** keep the existing clear, comment-driven style — explain the *why*, not
  just the *what*.

## Submitting changes

1. Branch from `master`.
2. Make your change with a focused, descriptive commit.
3. Open a pull request describing what changed and why. Screenshots/sample output
   help for notebook or generation changes.

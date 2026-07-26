# Contributing to LLM Workout

Thanks for your interest in improving this project! It's an educational resource,
so contributions that make a concept **clearer**, **more correct**, or **easier to
run** are all hugely welcome.

## Project layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Self-contained teaching notebooks, grouped into numbered topic folders (`NN_topic/`), one concept each (`NN_topic.ipynb`). |
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

- **Notebook naming & numbering:** `NN_topic.ipynb`, zero-padded, in teaching
  order. The numbers follow [CURRICULUM.md](./CURRICULUM.md) — that file is the
  source of truth, not the other way around. **Inserting a module is not a
  one-file change:** renumber every downstream notebook to keep the sequence
  gapless, add the row to the CURRICULUM.md module→file map, and update stale
  cross-references (`Module X.Y`, `NBnn`, `notebooks/nn_`) across `notebooks/`,
  `docs/`, and the READMEs. Then verify with:

  ```bash
  python scripts/check_references.py
  ```

  It fails on any notebook path or `Module X.Y` that no longer resolves, and on
  gaps or duplicates in the numbering. CI runs it on every pull request.
- **Keep notebooks runnable top-to-bottom** with no hidden state. Execute a
  notebook to verify it runs, then **clear outputs before committing**
  (`jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb`) so diffs stay
  small and free of non-deterministic churn.
- **Style:** keep the existing clear, comment-driven style — explain the *why*, not
  just the *what*.

## Submitting changes

1. Branch from `master`.
2. Make your change with a focused, descriptive commit.
3. Open a pull request describing what changed and why. Screenshots/sample output
   help for notebook or generation changes.

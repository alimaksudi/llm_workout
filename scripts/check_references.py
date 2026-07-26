#!/usr/bin/env python3
"""Guard the curriculum's cross-references.

CURRICULUM.md declares itself the source of truth for sequencing, and inserting a
module means renumbering everything downstream (see CONTRIBUTING.md). That is easy
to get 90% right and leave a handful of stale pointers behind -- which is exactly
what happened before this check existed.

It verifies four things:

  1. Every notebook path mentioned anywhere in the repo actually exists.
  2. Appendix A of CURRICULUM.md lists every Part I notebook, and only real files.
  3. Every "Module X.Y" referenced in prose is a module Appendix A defines.
  4. Notebook file numbers are gapless and unique across the topic folders.

Run it directly (`python scripts/check_references.py`); exits non-zero on failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
CURRICULUM = ROOT / "CURRICULUM.md"

# Part II keeps its own blueprint and its own lettered module ids (A.1, D.6, ...),
# so Appendix A of CURRICULUM.md covers Part I only.
PART_II_DIR = "10_applied"

errors: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def searchable_files() -> list[Path]:
    """Every file whose prose can carry a cross-reference."""
    files = [p for p in ROOT.glob("*.md")]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted(NOTEBOOKS.glob("*.md"))
    files += sorted(NOTEBOOKS.glob("**/*.ipynb"))
    return files


def text_of(path: Path) -> str:
    """Prose content of a file -- for notebooks, just the cell sources."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return raw
    nb = json.loads(raw)
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def parse_appendix_a() -> dict[str, str]:
    """Module id -> declared file path, from the Appendix A table."""
    body = CURRICULUM.read_text(encoding="utf-8")
    start = body.find("## Appendix A")
    if start == -1:
        fail("CURRICULUM.md: Appendix A (module -> notebook file map) is missing.")
        return {}
    section = body[start:body.find("## Appendix B", start)]
    mapping: dict[str, str] = {}
    for mod, path in re.findall(r"^\|\s*(\d+\.\d+)\s*\|[^|]*\|\s*`([^`]+)`\s*\|",
                                section, re.MULTILINE):
        if mod in mapping:
            fail(f"Appendix A lists module {mod} twice.")
        mapping[mod] = path
    if not mapping:
        fail("CURRICULUM.md: Appendix A parsed to zero rows -- has its table format changed?")
    return mapping


def check_declared_paths(appendix: dict[str, str]) -> None:
    for mod, rel in appendix.items():
        if not (ROOT / rel).exists():
            fail(f"Appendix A: module {mod} points at '{rel}', which does not exist.")


def check_every_notebook_is_listed(appendix: dict[str, str]) -> None:
    declared = {p for p in appendix.values() if p.endswith(".ipynb")}
    on_disk = {
        str(p.relative_to(ROOT))
        for p in NOTEBOOKS.glob("**/*.ipynb")
        if PART_II_DIR not in p.parts
    }
    for missing in sorted(on_disk - declared):
        fail(f"Part I notebook '{missing}' exists but is not in Appendix A.")


def check_mentioned_paths() -> None:
    pattern = re.compile(r"notebooks/[0-9]{2}_[a-z_]+/[0-9]{2}_[a-z0-9_]+\.ipynb")
    bare = re.compile(r"(?<![/\w])([0-9]{2}_[a-z0-9_]+\.ipynb)")
    real_names = {p.name for p in NOTEBOOKS.glob("**/*.ipynb")}

    for f in searchable_files():
        body = text_of(f)
        rel = f.relative_to(ROOT)
        for path in set(pattern.findall(body)):
            if not (ROOT / path).exists():
                fail(f"{rel}: references '{path}', which does not exist.")
        # Bare filenames like "29_reading_a_real_llm.ipynb" -- the exact form that
        # went stale before. Check the basename resolves to some real notebook.
        for name in set(bare.findall(body)):
            if name not in real_names:
                fail(f"{rel}: references notebook '{name}', which does not exist.")


def check_module_references(appendix: dict[str, str]) -> None:
    known = set(appendix)
    pattern = re.compile(r"Modules?\s+(\d+\.\d+)")
    for f in searchable_files():
        rel = f.relative_to(ROOT)
        if rel.name in {"CURRICULUM.md", "CURRICULUM_PART_II.md"}:
            continue  # these define and discuss the numbering, including planned modules
        for mod in sorted(set(pattern.findall(text_of(f)))):
            if mod not in known:
                fail(f"{rel}: refers to 'Module {mod}', which Appendix A does not define.")


def check_numbering_is_gapless() -> None:
    nums: dict[int, str] = {}
    for p in sorted(NOTEBOOKS.glob("**/*.ipynb")):
        n = int(p.name[:2])
        if n in nums:
            fail(f"Duplicate notebook number {n:02d}: '{nums[n]}' and '{p.name}'.")
        nums[n] = p.name
    if not nums:
        fail("No notebooks found.")
        return
    expected = set(range(1, max(nums) + 1))
    for gap in sorted(expected - set(nums)):
        fail(f"Notebook numbering has a gap at {gap:02d} -- the sequence must be gapless.")


def main() -> int:
    appendix = parse_appendix_a()
    check_declared_paths(appendix)
    check_every_notebook_is_listed(appendix)
    check_mentioned_paths()
    check_module_references(appendix)
    check_numbering_is_gapless()

    if errors:
        print(f"{len(errors)} stale reference(s):\n")
        for e in errors:
            print(f"  - {e}")
        print("\nSee CONTRIBUTING.md: inserting or moving a module is not a one-file change.")
        return 1

    n = len(list(NOTEBOOKS.glob("**/*.ipynb")))
    print(f"All references resolve: {n} notebooks, {len(appendix)} modules mapped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

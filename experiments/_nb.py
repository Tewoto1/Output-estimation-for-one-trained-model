"""experiments/_nb.py -- shared helper for the notebook *builder* scripts.

Every build_*.py used to re-define its own md()/code()/save and its own
repo-locator bootstrap cell. They now all use this. Usage in a builder:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _nb import NotebookBuilder, BOOTSTRAP_CELL

    nb = NotebookBuilder()
    nb.md("# Title")
    nb.code(BOOTSTRAP_CELL)          # standard "find the repo root" cell
    ...
    nb.save(os.path.join(os.path.dirname(__file__), "my_notebook.ipynb"))
"""
import json


# The standard first code cell of every generated notebook: locate the repo root
# (local run, Colab + Drive, or clone from REPO_URL), chdir there, put it on
# sys.path. Edit it HERE, not in individual builders.
BOOTSTRAP_CELL = r"""
LOCAL_REPO_DIR = ""   # e.g. "/content/drive/MyDrive/One trained case"  (leave "" to auto-find)
REPO_URL       = ""   # set to a git URL to clone instead
import os, sys, subprocess
try:
    import google.colab; IN_COLAB = True
except Exception:
    IN_COLAB = False
_MARK = os.path.join("model", "mlp.py")
def _find_repo():
    if LOCAL_REPO_DIR and os.path.isfile(os.path.join(LOCAL_REPO_DIR, _MARK)):
        return LOCAL_REPO_DIR
    if REPO_URL:
        dest = "/content/one_trained_case" if IN_COLAB else os.path.abspath("./_repo_clone")
        if not os.path.isfile(os.path.join(dest, _MARK)):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
        return dest
    here = os.path.abspath(".")
    for _ in range(6):
        if os.path.isfile(os.path.join(here, _MARK)):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Set LOCAL_REPO_DIR or REPO_URL.")
REPO = _find_repo(); os.chdir(REPO)
if REPO not in sys.path: sys.path.insert(0, REPO)
print("REPO:", REPO, "| IN_COLAB:", IN_COLAB)
"""


class NotebookBuilder:
    """Collects markdown/code cells and writes a valid nbformat-4 notebook."""

    def __init__(self):
        self.cells = []

    def _id(self):
        return f"cell-{len(self.cells):02d}"

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "id": self._id(), "metadata": {},
                           "source": text.splitlines(keepends=True)})

    def code(self, text):
        self.cells.append({"cell_type": "code", "id": self._id(), "metadata": {},
                           "execution_count": None, "outputs": [],
                           "source": text.strip("\n").splitlines(keepends=True)})

    def save(self, path):
        nb = {"cells": self.cells,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                          "name": "python3"},
                           "language_info": {"name": "python", "version": "3.12"},
                           "colab": {"provenance": []}},
              "nbformat": 4, "nbformat_minor": 5}
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path, "with", len(self.cells), "cells")

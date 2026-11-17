"""Prepare a Databricks notebook session: install our pinned packages and Stockfish.

Each notebook on Databricks starts in a fresh environment. It already has PyTorch
(built for its GPU driver), pandas, matplotlib and Jupyter, but not the rest of
requirements.txt or Stockfish. On any other machine `setup_databricks()` does
nothing, because the README setup has already installed everything.

Only the standard library is imported here, since this runs before anything is installed.
"""

import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Databricks keeps its own versions of these: its PyTorch matches its GPU driver, and the
# others run the notebook itself. Everything that affects training or grading
# (transformers, trl, peft, datasets, accelerate, bitsandbytes, chess) gets our pinned version.
KEEP_DATABRICKS_VERSION = ["torch", "pandas", "matplotlib", "jupyter", "ipywidgets", "pytest"]

# The libraries whose version must match the pins. If one was imported before the
# install, the old version stays loaded until Python restarts.
PINNED_IMPORTS = ["transformers", "trl", "peft", "datasets", "accelerate", "chess"]

# The same Stockfish release (19) as on our other machines, so engine scores match.
STOCKFISH_URL = ("https://github.com/official-stockfish/Stockfish/releases/download/"
                 "sf_19/stockfish-linux-x86-64-universal.tar.gz")
STOCKFISH_DIR = Path("/tmp/stockfish")  # the machine's own disk: downloaded again in each new session
STOCKFISH_BINARY = STOCKFISH_DIR / "stockfish" / "stockfish-linux-x86-64-universal"


def on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ or Path("/Workspace").is_dir()


def setup_databricks():
    """On Databricks, install the pinned packages and Stockfish for this session. Elsewhere, do nothing."""
    if not on_databricks():
        return
    install_packages()
    install_stockfish()
    os.environ["STOCKFISH_PATH"] = str(STOCKFISH_BINARY)
    print(f"Ready: pinned packages installed, keeping Databricks' torch {version('torch')}; "
          f"Stockfish at {STOCKFISH_BINARY}")


def databricks_requirements():
    """The lines of requirements.txt to install on Databricks: all pins except KEEP_DATABRICKS_VERSION.

    `-e .` is skipped too: the notebook's setup cell puts src/ on the import path instead.
    """
    lines = []
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[=<>!~;\s\[]", line)[0].lower()
        if name not in KEEP_DATABRICKS_VERSION:
            lines.append(line)
    return lines


def install_packages():
    stale = [name for name in PINNED_IMPORTS if name in sys.modules]
    if stale:
        raise RuntimeError(
            f"{', '.join(stale)} were imported before this setup, so the wrong versions may be loaded. "
            "Run dbutils.library.restartPython(), then run this cell again before any other."
        )
    # Pin Databricks' own versions too, so pip can never replace them (it stops with an error instead).
    keep = []
    for name in KEEP_DATABRICKS_VERSION:
        try:
            keep.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           *databricks_requirements(), *keep])


def install_stockfish():
    if STOCKFISH_BINARY.exists():
        return
    STOCKFISH_DIR.mkdir(parents=True, exist_ok=True)
    archive, _ = urllib.request.urlretrieve(STOCKFISH_URL)
    with tarfile.open(archive) as tar:
        tar.extractall(STOCKFISH_DIR, filter="data")

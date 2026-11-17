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
import sysconfig
import tarfile
import urllib.request
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Databricks keeps its own versions of these: its PyTorch matches its GPU driver, and the
# others run the notebook itself. Everything that affects training or grading
# (transformers, trl, peft, datasets, accelerate, bitsandbytes, chess) gets our pinned version.
KEEP_DATABRICKS_VERSION = ["torch", "pandas", "matplotlib", "jupyter", "ipywidgets", "pytest"]

# Extra limits on Databricks only. Our datasets version needs pyarrow 24+, and Databricks'
# own Spark client (databricks-connect) needs pyarrow below 25.
DATABRICKS_ONLY = ["pyarrow>=24,<25"]

# The same Stockfish release (19) as on our other machines, so engine scores match.
STOCKFISH_URL = ("https://github.com/official-stockfish/Stockfish/releases/download/"
                 "sf_19/stockfish-linux-x86-64-universal.tar.gz")
STOCKFISH_DIR = Path("/tmp/stockfish")  # the machine's own disk: downloaded again in each new session
STOCKFISH_BINARY = STOCKFISH_DIR / "stockfish" / "stockfish-linux-x86-64-universal"  # engine.py looks here too


def on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ or Path("/Workspace").is_dir()


def setup_databricks():
    """On Databricks, install the pinned packages and Stockfish for this session. Elsewhere, do nothing.

    If the install changed any package, Python restarts once so that no old version stays
    loaded in memory. Everything set up here survives the restart.
    """
    if not on_databricks():
        return
    add_src_to_path()
    install_stockfish()
    changed = install_packages()
    print(f"Ready: pinned packages installed, keeping Databricks' torch {version('torch')}; "
          f"Stockfish at {STOCKFISH_BINARY}")
    if changed:
        print("Restarting Python so the new versions load. If the notebook stops here, click Run all again.")
        from databricks.sdk.runtime import dbutils
        dbutils.library.restartPython()


def databricks_requirements():
    """The lines of requirements.txt to install on Databricks: all pins except KEEP_DATABRICKS_VERSION.

    `-e .` is skipped too: add_src_to_path() makes our code importable instead.
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


def installed_versions():
    return {dist.metadata["Name"].lower(): dist.version for dist in distributions()}


def install_packages():
    """Install the pins with pip. Returns True if any installed package changed."""
    # Pin Databricks' own versions too, so pip can never replace them (it stops with an error instead).
    keep = []
    for name in KEEP_DATABRICKS_VERSION:
        try:
            keep.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            pass
    before = installed_versions()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           *databricks_requirements(), *DATABRICKS_ONLY, *keep])
    return installed_versions() != before


def add_src_to_path():
    """Make src/ importable for the rest of the session, including after a Python restart."""
    pth_file = Path(sysconfig.get_paths()["purelib"]) / "chesslm_src.pth"
    pth_file.write_text(str(REPO_ROOT / "src") + "\n")


def install_stockfish():
    if STOCKFISH_BINARY.exists():
        return
    STOCKFISH_DIR.mkdir(parents=True, exist_ok=True)
    archive, _ = urllib.request.urlretrieve(STOCKFISH_URL)
    with tarfile.open(archive) as tar:
        tar.extractall(STOCKFISH_DIR, filter="data")

from pathlib import Path

SSREGPROS_ROOT = Path(__file__).parent
PROJECT_ROOT = SSREGPROS_ROOT.parent
CACHE_ROOT = PROJECT_ROOT / "cache"
CACHE_ROOT.mkdir(exist_ok=True)
_DATASET_ROOT = PROJECT_ROOT / "datasets"
_DATASET_ROOT.mkdir(exist_ok=True)
RAW_DATA_ROOT = _DATASET_ROOT / "raw"
RAW_DATA_ROOT.mkdir(exist_ok=True)
PROCESSED_DATA_ROOT = _DATASET_ROOT / "preprocessed"
PROCESSED_DATA_ROOT.mkdir(exist_ok=True)

import git

try:
    LATEST_COMMIT_SHA = git.Repo(PROJECT_ROOT).head.commit.hexsha
except ValueError:
    LATEST_COMMIT_SHA = "fake-git-SHA1"

"""Single place for paths and constants. Override the data location with the
BIRDSET_DIR environment variable if your copy lives somewhere else."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("BIRDSET_DIR", Path.home() / "data" / "birdset" / "NES"))
DERIVED_DIR = DATA_DIR / "derived"          # manifest.parquet, features.parquet (not in git)

SITE = "NES"
SR = 32_000                                 # BirdSet native sample rate (Hz)
CLIP_S = 5.0                                # every example is a 5 second clip
N_FOLDS = 10
SEED = 90051                                # one seed for every random choice

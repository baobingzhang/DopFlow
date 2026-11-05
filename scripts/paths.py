"""Project root. Set DOPFLOW_ROOT to keep data, caches, runs and results elsewhere."""
import os
from pathlib import Path
EXP = Path(os.environ.get("DOPFLOW_ROOT", Path(__file__).resolve().parents[1]))

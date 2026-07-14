from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.complex_feature_pipeline import run_full_complex_pipeline


if __name__ == "__main__":
    result = run_full_complex_pipeline()
    print(json.dumps(result, indent=2))

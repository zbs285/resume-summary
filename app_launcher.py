from __future__ import annotations

import sys
from pathlib import Path


if sys.stdout is None or sys.stderr is None:
    log_path = Path("/tmp/resume-summary-app.log")
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_stream
    sys.stderr = log_stream

from resume_summarizer.ui import main


if __name__ == "__main__":
    raise SystemExit(main())

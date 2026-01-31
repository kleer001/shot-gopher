"""Shared job state management."""

import threading
from typing import Dict

active_jobs: Dict[str, dict] = {}
active_jobs_lock = threading.Lock()

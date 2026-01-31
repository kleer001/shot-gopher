"""Installation state management for resume/recovery.

This module provides persistent state tracking for installation progress,
allowing interrupted installations to be resumed.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from env_config import INSTALL_DIR

from .utils import print_warning


class InstallationStateManager:
    """Manages installation state for resume/recovery."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or INSTALL_DIR / "install_state.json"
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load installation state from file."""
        if not self.state_file.exists():
            return self._create_initial_state()

        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)
            initial_state = self._create_initial_state()
            for key, value in initial_state.items():
                if key not in loaded_state:
                    loaded_state[key] = value
            return loaded_state
        except (json.JSONDecodeError, IOError):
            print_warning(f"Could not load state from {self.state_file}, creating new state")
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial state structure."""
        return {
            "version": "1.0",
            "environment": None,
            "last_updated": None,
            "components": {},
            "checkpoints": {}
        }

    def save_state(self):
        """Save installation state to file."""
        self.state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write (write to temp, then rename)
        temp_file = self.state_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, indent=2, fp=f)
            temp_file.replace(self.state_file)
        except IOError as e:
            print_warning(f"Could not save state: {e}")

    def set_environment(self, env_name: str):
        """Set the conda environment name."""
        self.state["environment"] = env_name
        self.save_state()

    def mark_component_started(self, comp_id: str):
        """Mark component installation as started."""
        self.state["components"][comp_id] = {
            "status": "in_progress",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": None
        }
        self.save_state()

    def mark_component_completed(self, comp_id: str):
        """Mark component installation as completed."""
        if comp_id in self.state["components"]:
            self.state["components"][comp_id]["status"] = "completed"
            self.state["components"][comp_id]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            self.state["components"][comp_id] = {
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": None
            }
        self.save_state()

    def mark_component_failed(self, comp_id: str, error: str):
        """Mark component installation as failed."""
        self.state["components"][comp_id] = {
            "status": "failed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": error
        }
        self.save_state()

    def get_component_status(self, comp_id: str) -> Optional[str]:
        """Get status of a component.

        Returns:
            "completed", "in_progress", "failed", or None
        """
        if comp_id not in self.state["components"]:
            return None
        return self.state["components"][comp_id].get("status")

    def get_incomplete_components(self) -> List[str]:
        """Get list of components that are not completed."""
        incomplete = []
        for comp_id, info in self.state["components"].items():
            if info.get("status") != "completed":
                incomplete.append(comp_id)
        return incomplete

    def can_resume(self) -> bool:
        """Check if there's a resumable installation."""
        return len(self.get_incomplete_components()) > 0

    def mark_checkpoint_downloaded(self, comp_id: str, path: Path):
        """Mark checkpoint as downloaded."""
        self.state["checkpoints"][comp_id] = {
            "downloaded": True,
            "path": str(path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_state()

    def is_checkpoint_downloaded(self, comp_id: str) -> bool:
        """Check if checkpoint is already downloaded."""
        return self.state["checkpoints"].get(comp_id, {}).get("downloaded", False)

    def clear_state(self):
        """Clear installation state (for fresh start)."""
        self.state = self._create_initial_state()
        self.save_state()

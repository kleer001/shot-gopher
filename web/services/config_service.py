"""
Configuration service for pipeline settings.

Provides a single source of truth for pipeline configuration,
following the DRY principle and Open/Closed Principle.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache


class ConfigService:
    """Service for loading and providing pipeline configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the configuration service.

        Args:
            config_path: Path to configuration JSON file.
                        Defaults to web/config/pipeline_config.json
        """
        if config_path is None:
            # Default to config directory relative to this file
            web_dir = Path(__file__).parent.parent
            config_path = web_dir / "config" / "pipeline_config.json"

        self.config_path = config_path
        self._config = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file.

        Returns:
            Dictionary containing all configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration (cached).

        Returns:
            Complete configuration dictionary
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def get_stages(self) -> Dict[str, Dict[str, Any]]:
        """Get all stage configurations.

        Returns:
            Dictionary mapping stage IDs to stage metadata
        """
        return self.config.get("stages", {})

    def get_stage(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific stage.

        Args:
            stage_id: ID of the stage (e.g., 'depth', 'roto')

        Returns:
            Stage configuration dictionary or None if not found
        """
        return self.get_stages().get(stage_id)

    def get_stage_name(self, stage_id: str) -> str:
        """Get display name for a stage.

        Args:
            stage_id: ID of the stage

        Returns:
            Display name or stage_id if not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            return stage.get("name", stage_id)
        return stage_id

    def get_stage_output_dir(self, stage_id: str) -> Optional[str]:
        """Get output directory for a stage.

        Args:
            stage_id: ID of the stage

        Returns:
            Output directory path or None if not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            return stage.get("outputDir")
        return None

    def get_output_directories(self) -> List[str]:
        """Get list of all output directories.

        Returns:
            List of output directory paths
        """
        dirs = []
        for stage in self.get_stages().values():
            output_dir = stage.get("outputDir")
            if output_dir:
                dirs.append(output_dir)
        return dirs

    def get_stage_dependencies(self, stage_id: str) -> List[str]:
        """Get dependencies for a stage.

        Args:
            stage_id: ID of the stage

        Returns:
            List of stage IDs that this stage depends on
        """
        stage = self.get_stage(stage_id)
        if stage:
            return stage.get("dependencies", [])
        return []

    def get_stage_enables(self, stage_id: str) -> List[str]:
        """Get stages that are enabled by this stage.

        Args:
            stage_id: ID of the stage

        Returns:
            List of stage IDs that this stage enables
        """
        stage = self.get_stage(stage_id)
        if stage:
            return stage.get("enables", [])
        return []

    def get_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all preset configurations.

        Returns:
            Dictionary mapping preset IDs to preset metadata
        """
        return self.config.get("presets", {})

    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific preset.

        Args:
            preset_id: ID of the preset (e.g., 'quick', 'full')

        Returns:
            Preset configuration dictionary or None if not found
        """
        return self.get_presets().get(preset_id)

    def get_supported_video_formats(self) -> List[str]:
        """Get list of supported video file formats.

        Returns:
            List of file extensions (e.g., ['.mp4', '.mov'])
        """
        return self.config.get("supportedVideoFormats", [])

    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration.

        Returns:
            Dictionary with WebSocket settings
        """
        return self.config.get("websocket", {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration.

        Returns:
            Dictionary with UI settings
        """
        return self.config.get("ui", {})

    def estimate_processing_time(self, stage_ids: List[str], frame_count: int) -> float:
        """Estimate total processing time for given stages.

        Args:
            stage_ids: List of stage IDs to process
            frame_count: Number of frames to process

        Returns:
            Estimated time in seconds
        """
        total_time = 0.0
        for stage_id in stage_ids:
            stage = self.get_stage(stage_id)
            if stage:
                time_per_frame = stage.get("estimatedTimePerFrame", 0)
                total_time += time_per_frame * frame_count
        return total_time


# Global instance for easy import
_config_service = None

def get_config_service() -> ConfigService:
    """Get the global configuration service instance.

    Returns:
        ConfigService instance
    """
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service

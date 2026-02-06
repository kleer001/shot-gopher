"""Palette service for loading color themes from TOML files."""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path
from typing import Dict, List, Optional

PALETTES_DIR = Path(__file__).parent.parent / "palettes"

ICON_MAP = {
    "moon": "\U0001F319",
    "sun": "\u2600\uFE0F",
    "stars": "\U0001F30C",
    "tree": "\U0001F332",
    "window": "\U0001FA9F",
    "hotdog": "\U0001F32D",
    "cactus": "\U0001F335",
    "snowflake": "\u2744\uFE0F",
}


class PaletteService:
    """Service for loading and managing color palettes."""

    def __init__(self, palettes_dir: Optional[Path] = None):
        self._palettes_dir = palettes_dir or PALETTES_DIR
        self._cache: Optional[Dict[str, dict]] = None

    def _load_palette_file(self, filepath: Path) -> Optional[dict]:
        """Load a single palette from a TOML file."""
        try:
            with open(filepath, "rb") as f:
                data = tomllib.load(f)

            palette_info = data.get("palette", {})
            colors_raw = data.get("colors", {})

            colors = {f"--{key}": value for key, value in colors_raw.items()}

            icon_key = palette_info.get("icon", "")
            icon = ICON_MAP.get(icon_key, icon_key)

            return {
                "id": palette_info.get("id", filepath.stem),
                "name": palette_info.get("name", filepath.stem.title()),
                "icon": icon,
                "colors": colors,
            }
        except Exception as e:
            print(f"Failed to load palette {filepath}: {e}")
            return None

    def get_palettes(self, force_reload: bool = False) -> Dict[str, dict]:
        """Get all available palettes."""
        if self._cache is not None and not force_reload:
            return self._cache

        palettes = {}
        if self._palettes_dir.exists():
            for filepath in sorted(self._palettes_dir.glob("*.toml")):
                palette = self._load_palette_file(filepath)
                if palette:
                    palettes[palette["id"]] = palette

        self._cache = palettes
        return palettes

    def get_palette(self, palette_id: str) -> Optional[dict]:
        """Get a specific palette by ID."""
        palettes = self.get_palettes()
        return palettes.get(palette_id)

    def get_palette_list(self) -> List[dict]:
        """Get palettes as a list (for API responses)."""
        return list(self.get_palettes().values())

    def get_default_palette_id(self) -> str:
        """Get the default palette ID."""
        return "dark"


_palette_service: Optional[PaletteService] = None


def get_palette_service() -> PaletteService:
    """Get the singleton palette service instance."""
    global _palette_service
    if _palette_service is None:
        _palette_service = PaletteService()
    return _palette_service

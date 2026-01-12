"""Progress bar management for the installation wizard.

This module provides progress bar functionality with tqdm support
and fallback for environments without tqdm.
"""


class ProgressBarManager:
    """Manages progress bars with ncurses support and fallback.

    Note: Full ncurses implementation deferred for maintainability.
    Currently uses simple inline progress bars.
    """

    def __init__(self):
        self.use_tqdm = False

        # Check if tqdm is available for better progress bars
        try:
            __import__('tqdm')
            self.use_tqdm = True
        except ImportError:
            pass

    def create_progress_bar(self, total: int, desc: str = ""):
        """Create a progress bar.

        Args:
            total: Total number of items
            desc: Description

        Returns:
            Progress bar object or None
        """
        if self.use_tqdm:
            import tqdm
            return tqdm.tqdm(total=total, desc=desc, unit='B', unit_scale=True)
        return None

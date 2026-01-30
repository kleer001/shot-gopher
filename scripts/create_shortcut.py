#!/usr/bin/env python3
"""Create desktop shortcuts for Shot Gopher.

This script creates platform-appropriate shortcuts:
- Windows: Creates .lnk shortcut on Desktop and/or Start Menu
- macOS: Creates alias on Desktop and offers to add to Dock
- Linux: Creates .desktop file in applications menu

Usage:
    python scripts/create_shortcut.py              # Interactive mode
    python scripts/create_shortcut.py --desktop    # Desktop shortcut only
    python scripts/create_shortcut.py --menu       # Start Menu/Applications only
    python scripts/create_shortcut.py --all        # All locations
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.resolve()


def get_desktop_path() -> Path:
    """Get the user's Desktop path."""
    system = platform.system()

    if system == "Windows":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            return Path(userprofile) / "Desktop"
        return Path.home() / "Desktop"
    elif system == "Darwin":
        return Path.home() / "Desktop"
    else:
        # Linux - check XDG
        xdg_desktop = os.environ.get("XDG_DESKTOP_DIR")
        if xdg_desktop:
            return Path(xdg_desktop)
        return Path.home() / "Desktop"


def create_windows_shortcut(target_path: Path, shortcut_path: Path) -> bool:
    """Create a Windows .lnk shortcut using PowerShell."""
    # Escape paths for PowerShell (replace backslashes, escape quotes)
    target_str = str(target_path).replace("'", "''")
    shortcut_str = str(shortcut_path).replace("'", "''")
    workdir_str = str(target_path.parent).replace("'", "''")

    # PowerShell script to create shortcut (use single quotes for paths)
    ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{shortcut_str}')
$Shortcut.TargetPath = '{target_str}'
$Shortcut.WorkingDirectory = '{workdir_str}'
$Shortcut.Description = 'Launch Shot Gopher VFX Pipeline'
$Shortcut.Save()
'''

    result = subprocess.run(
        ["powershell", "-Command", ps_script],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def create_windows_start_menu_shortcut(target_path: Path) -> bool:
    """Create a Windows Start Menu shortcut."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return False

    start_menu = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if not start_menu.exists():
        return False

    shortcut_path = start_menu / "Shot Gopher.lnk"
    return create_windows_shortcut(target_path, shortcut_path)


def create_mac_desktop_alias(target_path: Path, alias_path: Path) -> bool:
    """Create a macOS alias on the Desktop."""
    # Escape paths for AppleScript (escape backslashes and quotes)
    target_str = str(target_path).replace('\\', '\\\\').replace('"', '\\"')
    parent_str = str(alias_path.parent).replace('\\', '\\\\').replace('"', '\\"')
    name_str = alias_path.name.replace('\\', '\\\\').replace('"', '\\"')

    # Use osascript to create alias
    script = f'''
tell application "Finder"
    make new alias file at POSIX file "{parent_str}" to POSIX file "{target_str}"
    set name of result to "{name_str}"
end tell
'''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def add_mac_to_dock(app_path: Path) -> bool:
    """Add an item to the macOS Dock (optional, asks user first)."""
    # Escape path for XML (used in plist format)
    path_str = str(app_path).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # This adds a persistent Dock item using defaults
    # Note: This requires the full path and works with .command files
    script = f'''
do shell script "defaults write com.apple.dock persistent-apps -array-add '<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>{path_str}</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>'"

do shell script "killall Dock"
'''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def create_linux_desktop_entry(target_path: Path, create_desktop: bool = True, create_menu: bool = True) -> bool:
    """Create a Linux .desktop file.

    Args:
        target_path: Path to the launcher script
        create_desktop: Whether to create desktop shortcut
        create_menu: Whether to create applications menu entry
    """
    # Escape path for shell (single quotes need special handling)
    parent_escaped = str(target_path.parent).replace("'", "'\\''")
    name_escaped = target_path.name.replace("'", "'\\''")

    desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Shot Gopher
Comment=VFX Pipeline Web Interface
Exec=bash -c 'cd '\\'{parent_escaped}'\\' && ./'\\''{name_escaped}'\\'''
Icon=video-x-generic
Terminal=true
Categories=Graphics;Video;
"""

    created = False

    # Create in applications menu
    if create_menu:
        applications_dir = Path.home() / ".local" / "share" / "applications"
        applications_dir.mkdir(parents=True, exist_ok=True)
        desktop_file = applications_dir / "shot-gopher.desktop"
        desktop_file.write_text(desktop_entry, encoding='utf-8')
        desktop_file.chmod(0o755)
        created = True

    # Create on Desktop if requested and exists
    if create_desktop:
        desktop_path = get_desktop_path()
        if desktop_path.exists():
            desktop_shortcut = desktop_path / "Shot Gopher.desktop"
            desktop_shortcut.write_text(desktop_entry, encoding='utf-8')
            desktop_shortcut.chmod(0o755)
            created = True

    return created


def main():
    parser = argparse.ArgumentParser(description="Create Shot Gopher desktop shortcuts")
    parser.add_argument("--desktop", action="store_true", help="Create Desktop shortcut only")
    parser.add_argument("--menu", action="store_true", help="Create Start Menu/Applications shortcut only")
    parser.add_argument("--all", action="store_true", help="Create all shortcuts")
    parser.add_argument("--quiet", "-q", action="store_true", help="Non-interactive mode")
    args = parser.parse_args()

    repo_root = get_repo_root()
    system = platform.system()

    print()
    print("Shot Gopher - Shortcut Creator")
    print("=" * 35)
    print()

    # Determine which launcher to use
    if system == "Windows":
        launcher = repo_root / "Shot Gopher.bat"
        if not launcher.exists():
            print(f"ERROR: Launcher not found: {launcher}")
            sys.exit(1)
    elif system == "Darwin":
        launcher = repo_root / "Shot Gopher.command"
        if not launcher.exists():
            print(f"ERROR: Launcher not found: {launcher}")
            sys.exit(1)
    else:
        # Linux - use the .command file (it's a bash script)
        launcher = repo_root / "Shot Gopher.command"
        if not launcher.exists():
            print(f"ERROR: Launcher not found: {launcher}")
            sys.exit(1)

    created = []

    # Determine what to create
    create_desktop = args.desktop or args.all
    create_menu = args.menu or args.all

    # If no specific flags provided
    if not args.desktop and not args.menu and not args.all:
        if args.quiet:
            # Quiet mode with no flags: default to desktop only
            create_desktop = True
        else:
            # Interactive mode: ask user
            print(f"Platform: {system}")
            print(f"Launcher: {launcher}")
            print()

            response = input("Create Desktop shortcut? [Y/n]: ").strip().lower()
            create_desktop = response != 'n'

            if system == "Windows":
                response = input("Create Start Menu shortcut? [Y/n]: ").strip().lower()
                create_menu = response != 'n'
            elif system == "Darwin":
                response = input("Add to Dock? [y/N]: ").strip().lower()
                create_menu = response == 'y'
            else:
                # Linux - ask about applications menu
                response = input("Create applications menu entry? [Y/n]: ").strip().lower()
                create_menu = response != 'n'

    print()

    # Create shortcuts based on platform
    if system == "Windows":
        if create_desktop:
            desktop = get_desktop_path()
            shortcut_path = desktop / "Shot Gopher.lnk"
            print(f"Creating Desktop shortcut: {shortcut_path}")
            if create_windows_shortcut(launcher, shortcut_path):
                created.append("Desktop")
                print("  OK")
            else:
                print("  FAILED")

        if create_menu:
            print("Creating Start Menu shortcut...")
            if create_windows_start_menu_shortcut(launcher):
                created.append("Start Menu")
                print("  OK")
            else:
                print("  FAILED")

    elif system == "Darwin":
        if create_desktop:
            desktop = get_desktop_path()
            alias_path = desktop / "Shot Gopher"
            print(f"Creating Desktop alias: {alias_path}")
            if create_mac_desktop_alias(launcher, alias_path):
                created.append("Desktop")
                print("  OK")
            else:
                print("  FAILED (you can drag the .command file to Desktop manually)")

        if create_menu:
            print("Adding to Dock...")
            if add_mac_to_dock(launcher):
                created.append("Dock")
                print("  OK")
            else:
                print("  FAILED (you can drag the .command file to Dock manually)")

    else:  # Linux
        if create_menu:
            print("Creating applications menu entry...")
            if create_linux_desktop_entry(launcher, create_desktop=False, create_menu=True):
                created.append("Applications Menu")
                print("  OK")
            else:
                print("  FAILED")

        if create_desktop:
            print("Creating Desktop shortcut...")
            if create_linux_desktop_entry(launcher, create_desktop=True, create_menu=False):
                created.append("Desktop")
                print("  OK")
            else:
                print("  FAILED")

    print()
    if created:
        print(f"Created shortcuts: {', '.join(created)}")
        print()
        print("You can now launch Shot Gopher from:")
        for location in created:
            print(f"  - {location}")
    else:
        print("No shortcuts were created.")
        print()
        print("You can manually run the launcher:")
        print(f"  {launcher}")

    print()


if __name__ == "__main__":
    main()

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
        return Path(os.environ.get("USERPROFILE", "")) / "Desktop"
    elif system == "Darwin":
        return Path.home() / "Desktop"
    else:
        # Linux - check XDG
        xdg_desktop = os.environ.get("XDG_DESKTOP_DIR")
        if xdg_desktop:
            return Path(xdg_desktop)
        return Path.home() / "Desktop"


def create_windows_shortcut(target_path: Path, shortcut_path: Path, icon_path: Path = None):
    """Create a Windows .lnk shortcut using PowerShell."""

    # PowerShell script to create shortcut
    ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target_path}"
$Shortcut.WorkingDirectory = "{target_path.parent}"
$Shortcut.Description = "Launch Shot Gopher VFX Pipeline"
$Shortcut.Save()
'''

    result = subprocess.run(
        ["powershell", "-Command", ps_script],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def create_windows_start_menu_shortcut(target_path: Path):
    """Create a Windows Start Menu shortcut."""
    start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"

    if not start_menu.exists():
        return False

    shortcut_path = start_menu / "Shot Gopher.lnk"
    return create_windows_shortcut(target_path, shortcut_path)


def create_mac_desktop_alias(target_path: Path, alias_path: Path):
    """Create a macOS alias on the Desktop."""

    # Use osascript to create alias
    script = f'''
tell application "Finder"
    make new alias file at POSIX file "{alias_path.parent}" to POSIX file "{target_path}"
    set name of result to "{alias_path.name}"
end tell
'''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def add_mac_to_dock(app_path: Path):
    """Add an item to the macOS Dock (optional, asks user first)."""

    # This adds a persistent Dock item using defaults
    # Note: This requires the full path and works with .command files
    script = f'''
tell application "System Events"
    tell dock preferences
        set properties to {{animate:true}}
    end tell
end tell

tell application "Dock"
    quit
end tell

delay 1

do shell script "defaults write com.apple.dock persistent-apps -array-add '<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>{app_path}</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>'"

do shell script "killall Dock"
'''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )

    return result.returncode == 0


def create_linux_desktop_entry(target_path: Path):
    """Create a Linux .desktop file."""

    # Create in user's applications directory
    applications_dir = Path.home() / ".local" / "share" / "applications"
    applications_dir.mkdir(parents=True, exist_ok=True)

    desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Shot Gopher
Comment=VFX Pipeline Web Interface
Exec=bash -c 'cd "{target_path.parent}" && ./"{target_path.name}"'
Icon=video-x-generic
Terminal=true
Categories=Graphics;Video;
"""

    desktop_file = applications_dir / "shot-gopher.desktop"
    desktop_file.write_text(desktop_entry)
    desktop_file.chmod(0o755)

    # Also create on Desktop if it exists
    desktop_path = get_desktop_path()
    if desktop_path.exists():
        desktop_shortcut = desktop_path / "Shot Gopher.desktop"
        desktop_shortcut.write_text(desktop_entry)
        desktop_shortcut.chmod(0o755)

    return True


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

    # If no flags, ask interactively
    if not args.desktop and not args.menu and not args.all and not args.quiet:
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
        print("Creating Linux desktop entry...")
        if create_linux_desktop_entry(launcher):
            created.append("Applications Menu")
            if get_desktop_path().exists():
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

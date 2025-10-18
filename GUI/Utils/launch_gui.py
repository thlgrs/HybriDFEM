#!/usr/bin/env python3
"""
HybridFEM GUI Launcher
Handles path setup and launches the GUI application
"""

import sys
from pathlib import Path


def setup_paths():
    """Add necessary paths to Python path"""

    # Get the script directory
    script_dir = Path(__file__).parent.absolute()

    # Add Theo directory to path (adjust as needed)
    theo_path = script_dir / "Theo"
    if theo_path.exists():
        sys.path.insert(0, str(theo_path.parent))
        print(f"Added to path: {theo_path.parent}")
    else:
        # Try to find Theo in parent directories
        current = script_dir
        for _ in range(3):  # Search up to 3 levels
            current = current.parent
            theo_candidate = current / "Theo"
            if theo_candidate.exists():
                sys.path.insert(0, str(current))
                print(f"Added to path: {current}")
                break
        else:
            print("Warning: Could not find Theo directory")
            print("Please ensure your HybridFEM code is in the correct location")

    # Add Legacy directory if it exists
    legacy_path = script_dir / "Legacy"
    if legacy_path.exists():
        sys.path.insert(0, str(legacy_path.parent))
        print(f"Added to path: {legacy_path.parent}")


def check_dependencies():
    """Check if required packages are installed"""

    required_packages = [
        ('PyQt6', 'PyQt6'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
    ]

    optional_packages = [
        ('rhino3dm', 'rhino3dm (for .3dm file support)'),
        ('gmsh', 'gmsh (for mesh generation)'),
        ('meshio', 'meshio (for mesh handling)'),
    ]

    missing_required = []
    missing_optional = []

    # Check required packages
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} found")
        except ImportError:
            missing_required.append(name)
            print(f"✗ {name} NOT found")

    # Check optional packages
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✓ {name} found")
        except ImportError:
            missing_optional.append(name)
            print(f"○ {name} not found (optional)")

    if missing_required:
        print("\n" + "=" * 60)
        print("ERROR: Missing required packages:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        print("=" * 60)
        return False

    if missing_optional:
        print("\n" + "=" * 60)
        print("Optional packages not found:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        print("=" * 60)

    return True


def main():
    """Launch the GUI application"""

    print("=" * 60)
    print("HybridFEM GUI Launcher")
    print("=" * 60)
    print()

    # Setup paths
    print("Setting up paths...")
    setup_paths()
    print()

    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nCannot launch GUI due to missing dependencies")
        sys.exit(1)
    print()

    # Import and launch GUI
    try:
        print("Launching HybridFEM GUI...")
        print("=" * 60)
        print()

        # Import the GUI module
        from MainWindow import main as gui_main

        # Launch the application
        gui_main()

    except ImportError as e:
        print(f"\nError importing GUI module: {e}")
        print("\nMake sure MainWindow.py is in the current directory")
        sys.exit(1)

    except Exception as e:
        print(f"\nError launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

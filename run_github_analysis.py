#!/usr/bin/env python3
"""Runner script for GitHub issues analysis."""

from src.analysis.github_issues_analyzer import main
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import after path is set

if __name__ == "__main__":
    main()

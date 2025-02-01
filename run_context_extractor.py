#!/usr/bin/env python3
"""Runner script for the context extractor."""

from src.utils.context_extractor import main
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Now import after path is set

if __name__ == "__main__":
    main()

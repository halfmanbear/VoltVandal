#!/usr/bin/env python3
import sys
import os

# Add src to sys.path so we can import the package without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from voltvandal.main import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

if __name__ != "__main__":
    # Imported as `voltvandal` from this folder (e.g. `python -m voltvandal.main`):
    # this file shadows the real package, so act as that package instead.
    __path__ = [os.path.join(SRC, "voltvandal")]
else:
    # Add src to sys.path so we can import the package without installation
    sys.path.insert(0, SRC)

    from voltvandal.main import main

    raise SystemExit(main())

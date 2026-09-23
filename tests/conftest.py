import os
import sys

# Allow running the tests from a plain checkout (reference backend only) as well as
# against an installed wheel that includes the native extension.
try:
    import flipster  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

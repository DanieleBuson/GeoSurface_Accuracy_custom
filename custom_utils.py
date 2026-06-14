"""
Re-export of files_utils.py for the custom Trentino dataset pipeline.

This module exists for backward compatibility with the existing notebooks/scripts,
which import from `custom_utils`. The implementation itself (readers, IDW,
plotting, boundary-overlap) lives in `files_utils.py`, copied into this folder so
GeoSurface_Accuracy_custom is self-contained and can be cloned/pushed on its own.
"""
from files_utils import *  # noqa: F401,F403

from pathlib import Path
import sys


def default_model_bundle():
    if getattr(sys, 'frozen', False):
        executable = Path(sys.executable).resolve()
        if sys.platform == 'darwin':
            return executable.parent.parent / 'Resources' / 'model_bundle'
        return executable.parent / 'model_bundle'
    return Path(__file__).resolve().parent / 'model_bundle'

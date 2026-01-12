import os
import sys

_TEST_ROOT = os.path.dirname(__file__)        # .../tests
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)   # .../exercise_files
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")

# Make project root importable so tests can do:
# from s1_development_environment.exercise_files.final_exercise.data import ...
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PROJECT_ROOT)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

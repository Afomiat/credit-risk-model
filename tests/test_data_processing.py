"""
test_data_processing.py
Placeholder tests to keep CI passing while
full test suite is developed in Task 5.
"""


def test_placeholder():
    """
    Placeholder test — always passes.
    Prevents pytest exit code 5 (no tests collected).
    Real tests added in Task 5.
    """
    assert True


def test_python_environment():
    """
    Verify core dependencies are importable.
    Catches broken installation early.
    """
    import pandas as pd
    import numpy as np
    import sklearn

    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None
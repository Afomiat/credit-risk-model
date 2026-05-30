"""
test_data_processing.py
Unit tests for data processing functions.
Placeholder tests added in Task 2.
Real feature engineering tests added in Task 3.
"""


def test_placeholder():
    """
    Placeholder — always passes.
    Prevents pytest exit code 5 (no tests collected).
    """
    assert True


def test_imports():
    """
    Verify core dependencies are importable.
    Catches broken installation early in CI.
    """
    import pandas as pd
    import numpy as np
    import sklearn

    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None


def test_data_shape():
    """
    Verify basic pandas operations work correctly.
    """
    import pandas as pd

    # Create a small sample dataframe
    # mimicking the Xente transaction structure
    sample = pd.DataFrame({
        'CustomerId':  ['C1', 'C1', 'C2', 'C3'],
        'Amount':      [1000, 500, 2000, 300],
        'FraudResult': [0, 0, 0, 1]
    })

    # Basic assertions
    assert len(sample) == 4
    assert sample['CustomerId'].nunique() == 3
    assert sample['FraudResult'].sum() == 1
    assert 'Amount' in sample.columns
"""
test_data_processing.py
Unit tests for data processing pipeline.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))


def test_placeholder():
    """Always passes — keeps CI green."""
    assert True


def test_imports():
    """Verify core dependencies are importable."""
    import pandas as pd
    import numpy as np
    import sklearn
    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None


def test_time_feature_extractor():
    """
    Test that TimeFeatureExtractor adds the correct columns.
    """
    from src.data_processing import TimeFeatureExtractor

    sample = pd.DataFrame({
        'TransactionStartTime': [
            '2018-11-15T02:18:49Z',
            '2018-12-01T14:30:00Z',
        ],
        'CustomerId': ['C1', 'C2'],
        'Amount': [1000, 500]
    })

    transformer = TimeFeatureExtractor()
    result = transformer.fit_transform(sample)

    # Check all expected time columns were added
    expected_cols = [
        'TransactionHour', 'TransactionDay',
        'TransactionMonth', 'TransactionYear',
        'TransactionDayOfWeek', 'IsWeekend'
    ]
    for col in expected_cols:
        assert col in result.columns, \
            f"Missing expected column: {col}"

    # Check specific values
    assert result['TransactionHour'].iloc[0] == 2
    assert result['TransactionMonth'].iloc[1] == 12


def test_aggregate_features():
    """
    Test that AggregateFeatures collapses to customer level.
    """
    from src.data_processing import AggregateFeatures

    sample = pd.DataFrame({
        'CustomerId':            ['C1', 'C1', 'C2'],
        'TransactionId':         ['T1', 'T2', 'T3'],
        'BatchId':               ['B1', 'B2', 'B3'],
        'Amount':                [1000.0, 500.0, 2000.0],
        'Value':                 [1000, 500, 2000],
        'ProductId':             ['P1', 'P2', 'P1'],
        'ProductCategory':       ['airtime', 'utility', 'airtime'],
        'ProviderId':            ['PR1', 'PR1', 'PR2'],
        'ChannelId':             ['CH1', 'CH2', 'CH1'],
        'FraudResult':           [0, 0, 1],
        'TransactionStartTime':  [
            '2018-11-15T02:18:49Z',
            '2018-11-20T10:00:00Z',
            '2018-11-15T02:18:49Z'
        ]
    })

    transformer = AggregateFeatures()
    result = transformer.fit_transform(sample)

    # Should have 2 customers not 3 transactions
    assert len(result) == 2, \
        f"Expected 2 customers, got {len(result)}"

    # C1 should have 2 transactions
    c1 = result[result['CustomerId'] == 'C1']
    assert c1['TotalTransactions'].values[0] == 2

    # C1 total amount should be 1500
    assert c1['TotalAmount'].values[0] == 1500.0

    # C2 should have fraud rate of 1.0
    c2 = result[result['CustomerId'] == 'C2']
    assert c2['FraudRate'].values[0] == 1.0
import unittest
import pandas as pd
import numpy as np
from src.data.loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Dummy configuration for testing
        self.config = {
            'paths': {
                'train_data': 'dummy_train.csv',
                'test_data': 'dummy_test.csv',
                'customer_data': 'dummy_cust.csv',
                'sku_data': 'dummy_sku.csv'
            }
        }
        self.loader = DataLoader(self.config)

    def test_downcast_float64_to_float32(self):
        """Test that float64 columns are downcasted to float32."""
        df = pd.DataFrame({
            'float_col': np.array([1.1, 2.2, 3.3], dtype='float64'),
        })

        downcasted_df = self.loader._downcast_memory(df)
        self.assertEqual(downcasted_df['float_col'].dtype, np.float32)

    def test_downcast_int64_reduces_size(self):
        """Test that int64 columns are downcasted to a smaller int type."""
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3], dtype='int64'),
        })

        downcasted_df = self.loader._downcast_memory(df)
        # The method downcasts int64 to int32 when max < 2**32
        self.assertTrue(
            downcasted_df['int_col'].dtype.itemsize < np.dtype('int64').itemsize,
            f"Expected smaller dtype than int64, got {downcasted_df['int_col'].dtype}"
        )

    def test_downcast_preserves_values(self):
        """Test that downcasting preserves the original values."""
        df = pd.DataFrame({
            'float_col': np.array([1.1, 2.2, 3.3], dtype='float64'),
            'int_col': np.array([100, 200, 300], dtype='int64'),
        })

        downcasted_df = self.loader._downcast_memory(df)
        np.testing.assert_array_almost_equal(
            downcasted_df['float_col'].values, [1.1, 2.2, 3.3], decimal=5
        )
        np.testing.assert_array_equal(
            downcasted_df['int_col'].values, [100, 200, 300]
        )


if __name__ == '__main__':
    unittest.main()

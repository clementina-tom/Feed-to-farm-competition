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

    def test_downcast_memory(self):
        """Test that float64 and int64 are explicitly downcasted to save memory."""
        df = pd.DataFrame({
            'float_col': np.array([1.1, 2.2, 3.3], dtype='float64'),
            'int_col': np.array([1, 2, 3], dtype='int64')
        })
        
        downcasted_df = self.loader._downcast_memory(df)
        
        self.assertEqual(downcasted_df['float_col'].dtype, 'float32')
        self.assertEqual(downcasted_df['int_col'].dtype, 'int32')

if __name__ == '__main__':
    unittest.main()

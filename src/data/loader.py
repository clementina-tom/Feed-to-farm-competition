import pandas as pd
import numpy as np
import logging

class DataLoader:
    def __init__(self, config):
        """
        Initializes the DataLoader with the given configuration dictionary.
        """
        self.config = config
        self.paths = config['paths']
        self.logger = logging.getLogger(self.__class__.__name__)

    def _downcast_memory(self, df):
        """
        Downcasts numerical columns to lighter data types to save memory.
        """
        self.logger.info("Downcasting memory footprint...")
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].max() < 2**32:
                df[col] = df[col].astype('int32')
        return df

    def load_all(self):
        """
        Loads train, test, customer, and sku datasets, applies downcasting,
        and parses datetime columns.
        Returns:
            train, test, customer, sku (pd.DataFrame)
        """
        self.logger.info(f"Loading Train data from {self.paths['train_data']}")
        train = pd.read_csv(self.paths['train_data'])
        
        self.logger.info(f"Loading Test data from {self.paths['test_data']}")
        test = pd.read_csv(self.paths['test_data'])
        
        self.logger.info(f"Loading Customer data from {self.paths['customer_data']}")
        customer = pd.read_csv(self.paths['customer_data'])
        
        self.logger.info(f"Loading SKU data from {self.paths['sku_data']}")
        sku = pd.read_csv(self.paths['sku_data'])

        # Downcast
        train = self._downcast_memory(train)
        test = self._downcast_memory(test)
        customer = self._downcast_memory(customer)
        sku = self._downcast_memory(sku)

        # Datetime Parsing
        self.logger.info("Parsing datetime columns...")
        train["week_start"] = pd.to_datetime(train["week_start"])
        test["week_start"] = pd.to_datetime(test["week_start"])
        customer["customer_created_at"] = pd.to_datetime(customer["customer_created_at"])

        return train, test, customer, sku

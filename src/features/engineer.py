import pandas as pd
import numpy as np
import logging
import gc
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.le = LabelEncoder()

    def _weeks_since_last_purchase(self, x):
        out, last = [], None
        for i, v in enumerate(x):
            if v > 0:
                last = i
                out.append(0)
            else:
                out.append(i - last if last is not None else np.nan)
        return pd.Series(out, index=x.index)

    def engineer_features(self, train, test):
        self.logger.info("Creating base universal dataframe...")
        base_cols = ["customer_id", "product_unit_variant_id", "week_start", "qty_this_week"]
        
        # In test, we don't have qty_this_week yet, assign 0.0 for rolling window continuity
        test_dummy = test[["customer_id", "product_unit_variant_id", "week_start"]].copy()
        test_dummy["qty_this_week"] = 0.0
        
        temp_df = pd.concat([train[base_cols], test_dummy], ignore_index=True)
        temp_df = temp_df.sort_values(["customer_id", "product_unit_variant_id", "week_start"]).reset_index(drop=True)
        
        pair_grp = temp_df.groupby(["customer_id", "product_unit_variant_id"])["qty_this_week"]
        
        self.logger.info("Generating Pair Features (Lags, Roll, Recency)...")
        temp_df["lag1"] = pair_grp.shift(1)
        temp_df["lag2"] = pair_grp.shift(2)
        temp_df["roll_mean_4"] = pair_grp.transform(lambda x: x.shift(1).rolling(4).mean())
        temp_df["is_new_pair"] = pair_grp.shift(1).isna().astype(int)
        temp_df["pair_buy_rate"] = pair_grp.transform(lambda x: x.shift(1).expanding().mean())
        temp_df["pair_recency"] = pair_grp.transform(lambda x: self._weeks_since_last_purchase(x).shift(1))
        
        self.logger.info("Generating Customer Momentum...")
        cust_grp = temp_df.groupby("customer_id")["qty_this_week"]
        temp_df["cust_lag1"] = cust_grp.shift(1)
        temp_df["cust_roll_4"] = cust_grp.transform(lambda x: x.shift(1).rolling(4).mean())
        
        self.logger.info("Generating Global Product Trends...")
        global_weekly = (
            temp_df.groupby(["product_unit_variant_id", "week_start"])["qty_this_week"]
            .sum()
            .reset_index()
            .rename(columns={"qty_this_week": "global_weekly_vol"})
        )
        temp_df = temp_df.merge(global_weekly, on=["product_unit_variant_id", "week_start"], how="left")
        prod_grp = temp_df.groupby("product_unit_variant_id")["global_weekly_vol"]
        temp_df["global_lag1"] = prod_grp.shift(1)
        temp_df["global_roll_4"] = prod_grp.transform(lambda x: x.shift(1).rolling(4).mean())
        
        self.logger.info("Generating Seasonality...")
        temp_df["month"] = temp_df["week_start"].dt.month.fillna(0).astype(int)
        temp_df["week_of_year"] = temp_df["week_start"].dt.isocalendar().week.fillna(0).astype(int)
        
        feature_cols = [
            "lag1", "lag2", "roll_mean_4",
            "cust_lag1", "cust_roll_4",
            "global_lag1", "global_roll_4",
            "pair_buy_rate", "pair_recency",
            "is_new_pair", "month", "week_of_year",
        ]
        
        self.logger.info("Merging features back to train and test...")
        merge_cols = ["customer_id", "product_unit_variant_id", "week_start"]
        train = train.merge(temp_df[merge_cols + feature_cols], on=merge_cols, how="left")
        test = test.merge(temp_df[merge_cols + feature_cols], on=merge_cols, how="left")
        
        # Cleanup
        del temp_df, pair_grp, cust_grp, prod_grp, global_weekly
        gc.collect()
        
        return train, test, feature_cols

    def preprocess_metadata(self, train, test, customer, sku, feature_cols):
        self.logger.info("Merging Customer and SKU Metadata...")
        train = train.merge(customer, on="customer_id", how="left")
        train = train.merge(sku, on="product_unit_variant_id", how="left")
        
        test = test.merge(customer, on="customer_id", how="left")
        test = test.merge(sku, on="product_unit_variant_id", how="left")
        
        cat_candidates = ["customer_category", "customer_status", "grade_name", "unit_name"]
        cat_cols = []
        for c in cat_candidates:
            if f"{c}_x" in train.columns:
                cat_cols.append(f"{c}_x")
            elif c in train.columns:
                cat_cols.append(c)
                
        self.logger.info("Encoding Categorical Variables...")
        for col in cat_cols:
            train[col] = train[col].astype(str).fillna("UNKNOWN")
            test[col] = test[col].astype(str).fillna("UNKNOWN")
            self.le.fit(pd.concat([train[col], test[col]]))
            train[col] = self.le.transform(train[col])
            test[col] = self.le.transform(test[col])
            
        self.logger.info("Filling missing numerical values...")
        for col in feature_cols:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
            
        return train, test, cat_cols

    def generate_targets(self, train):
        self.logger.info("Generating Training Targets (1 week & 2 week)...")
        train = train.sort_values(["customer_id", "product_unit_variant_id", "week_start"])
        grp = train.groupby(["customer_id", "product_unit_variant_id"])["qty_this_week"]
        
        train["target_qty_1w"] = grp.shift(-1).fillna(0)
        train["target_qty_2w"] = grp.shift(-2).fillna(0)
        
        train["target_buy_1w"] = (train["target_qty_1w"] > 0).astype(int)
        train["target_buy_2w"] = (train["target_qty_2w"] > 0).astype(int)
        
        return train

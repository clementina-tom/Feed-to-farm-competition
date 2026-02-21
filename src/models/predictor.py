import pandas as pd
import numpy as np
import logging
import os

class ModelPredictor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Scaling parameters
        self.scale_p1 = config['scaling']['purchase_1w_scale']
        self.scale_p2 = config['scaling']['purchase_2w_scale']
        self.thresh_q1 = config['scaling']['qty_1w_threshold']
        self.thresh_q2 = config['scaling']['qty_2w_threshold']
        
        # Ensemble weights
        self.w_lgb = config['ensemble']['lgbm_weight']
        self.w_cb = config['ensemble']['catboost_weight']

    def predict(self, models, test, features):
        """
        Runs inference across all seeds and both model types,
        then blends and applies decoupled post-processing.
        Args:
            models (dict): Dict of model lists from ModelTrainer.
            test (pd.DataFrame): The test dataframe with all feature columns.
            features (list): The list of all feature column names used in training.
        Returns:
            pd.DataFrame: The final submission dataframe.
        """
        self.logger.info("Starting Ensemble Inference...")
        
        num_seeds = len(self.config['model']['seeds'])
        
        lgb_p1_all, lgb_p2_all = [], []
        lgb_q1_all, lgb_q2_all = [], []
        
        cb_p1_all, cb_p2_all = [], []
        cb_q1_all, cb_q2_all = [], []
        
        X_test = test[features]

        self.logger.info(f"Predicting across {num_seeds} seeds...")
        for i in range(num_seeds):
            # LightGBM Predictions
            lgb_p1_all.append(models['lgb_clf1'][i].predict_proba(X_test)[:, 1])
            lgb_p2_all.append(models['lgb_clf2'][i].predict_proba(X_test)[:, 1])
            lgb_q1_all.append(np.maximum(0, models['lgb_reg1'][i].predict(X_test)))
            lgb_q2_all.append(np.maximum(0, models['lgb_reg2'][i].predict(X_test)))
            
            # CatBoost Predictions
            cb_p1_all.append(models['cb_clf1'][i].predict_proba(X_test)[:, 1])
            cb_p2_all.append(models['cb_clf2'][i].predict_proba(X_test)[:, 1])
            cb_q1_all.append(np.maximum(0, models['cb_reg1'][i].predict(X_test)))
            cb_q2_all.append(np.maximum(0, models['cb_reg2'][i].predict(X_test)))

        self.logger.info("Averaging seed predictions...")
        lgb_p1 = np.mean(lgb_p1_all, axis=0)
        lgb_p2 = np.mean(lgb_p2_all, axis=0)
        lgb_q1 = np.mean(lgb_q1_all, axis=0)
        lgb_q2 = np.mean(lgb_q2_all, axis=0)
        
        cb_p1 = np.mean(cb_p1_all, axis=0)
        cb_p2 = np.mean(cb_p2_all, axis=0)
        cb_q1 = np.mean(cb_q1_all, axis=0)
        cb_q2 = np.mean(cb_q2_all, axis=0)

        self.logger.info(f"Blending LGBM ({self.w_lgb}) and CatBoost ({self.w_cb})...")
        raw_p1 = (lgb_p1 * self.w_lgb) + (cb_p1 * self.w_cb)
        raw_p2 = (lgb_p2 * self.w_lgb) + (cb_p2 * self.w_cb)
        raw_q1 = (lgb_q1 * self.w_lgb) + (cb_q1 * self.w_cb)
        raw_q2 = (lgb_q2 * self.w_lgb) + (cb_q2 * self.w_cb)

        self.logger.info("Applying Decoupled Post-Processing...")
        submission = test[["ID"]].copy().reset_index(drop=True)

        # STEP A: Purchase probability — scale aggressively for AUC ranking
        submission["Target_purchase_next_1w"] = np.clip(raw_p1 * self.scale_p1, 0, 1)
        submission["Target_purchase_next_2w"] = np.clip(raw_p2 * self.scale_p2, 0, 1)

        # STEP B: Quantity — expected value, forcing low-confidence rows to 0
        qty_1 = raw_p1 * raw_q1
        qty_1 = np.where(raw_p1 < self.thresh_q1, 0, qty_1)   # avoid indexing issues
        submission["Target_qty_next_1w"] = np.clip(qty_1, a_min=0, a_max=None)

        qty_2 = raw_p2 * raw_q2
        qty_2 = np.where(raw_p2 < self.thresh_q2, 0, qty_2)
        submission["Target_qty_next_2w"] = np.clip(qty_2, a_min=0, a_max=None)

        # Save submission
        out_path = self.config['paths']['submission_file']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        submission.to_csv(out_path, index=False)
        self.logger.info(f"✅ Submission saved to {out_path}")
        
        return submission

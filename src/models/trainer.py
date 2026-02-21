import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
import numpy as np
import logging
import gc
import joblib
import os

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.seeds = config['model']['seeds']
        
        self.lgb_params = {
            'n_estimators': config['model']['n_estimators'],
            'learning_rate': config['model']['learning_rate'],
            'num_leaves': config['model']['num_leaves'],
            'feature_fraction': config['model']['feature_fraction'],
            'bagging_fraction': config['model']['bagging_fraction'],
            'bagging_freq': config['model']['bagging_freq'],
            'verbose': -1
        }
        
        self.cb_params = {
            'iterations': config['catboost']['iterations'],
            'learning_rate': config['catboost']['learning_rate'],
            'depth': config['catboost']['depth'],
            'verbose': 0,
            'allow_writing_files': False
        }

        os.makedirs(config['paths']['model_dir'], exist_ok=True)

    def train_hybrid_ensemble(self, train, features, cat_cols):
        self.logger.info(f"Starting Hybrid Ensemble Training over {len(self.seeds)} seeds...")
        
        models = {
            'lgb_clf1': [], 'lgb_clf2': [], 'lgb_reg1': [], 'lgb_reg2': [],
            'cb_clf1': [], 'cb_clf2': [], 'cb_reg1': [], 'cb_reg2': []
        }
        
        # Prepare targets
        y_buy_1w = train["target_buy_1w"]
        y_buy_2w = train["target_buy_2w"]
        y_qty_1w = train["target_qty_1w"]
        y_qty_2w = train["target_qty_2w"]
        
        # Masks for Tweedie Regressors (Train only on positive quantities)
        mask_1w = y_buy_1w == 1
        mask_2w = y_buy_2w == 1
        
        X_train = train[features]
        X_train_pos_1w = train.loc[mask_1w, features]
        y_qty_1w_pos = y_qty_1w[mask_1w]
        
        X_train_pos_2w = train.loc[mask_2w, features]
        y_qty_2w_pos = y_qty_2w[mask_2w]

        # CatBoost needs categorical indices
        cat_indices = [features.index(c) for c in cat_cols]

        for seed in self.seeds:
            self.logger.info(f"--- Training Seed {seed} ---")
            
            # --- LightGBM ---
            lgb_p = self.lgb_params.copy()
            lgb_p['random_state'] = seed
            
            clf1 = lgb.LGBMClassifier(**lgb_p)
            clf2 = lgb.LGBMClassifier(**lgb_p)
            reg1 = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **lgb_p)
            reg2 = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.6, **lgb_p)
            
            clf1.fit(X_train, y_buy_1w)
            clf2.fit(X_train, y_buy_2w)
            reg1.fit(X_train_pos_1w, y_qty_1w_pos)
            reg2.fit(X_train_pos_2w, y_qty_2w_pos)
            
            models['lgb_clf1'].append(clf1)
            models['lgb_clf2'].append(clf2)
            models['lgb_reg1'].append(reg1)
            models['lgb_reg2'].append(reg2)

            # --- CatBoost ---
            cb_p = self.cb_params.copy()
            cb_p['random_seed'] = seed
            
            cb_clf1 = CatBoostClassifier(**cb_p, loss_function='Logloss')
            cb_clf2 = CatBoostClassifier(**cb_p, loss_function='Logloss')
            cb_reg1 = CatBoostRegressor(**cb_p, loss_function='Tweedie:variance_power=1.3')
            cb_reg2 = CatBoostRegressor(**cb_p, loss_function='Tweedie:variance_power=1.6')
            
            cb_clf1.fit(X_train, y_buy_1w, cat_features=cat_indices)
            cb_clf2.fit(X_train, y_buy_2w, cat_features=cat_indices)
            cb_reg1.fit(X_train_pos_1w, y_qty_1w_pos, cat_features=cat_indices)
            cb_reg2.fit(X_train_pos_2w, y_qty_2w_pos, cat_features=cat_indices)
            
            models['cb_clf1'].append(cb_clf1)
            models['cb_clf2'].append(cb_clf2)
            models['cb_reg1'].append(cb_reg1)
            models['cb_reg2'].append(cb_reg2)
            
            gc.collect()

        # Save the ensemble
        model_path = os.path.join(self.config['paths']['model_dir'], 'hybrid_ensemble.pkl')
        self.logger.info(f"Saving models to {model_path}")
        joblib.dump(models, model_path)
        
        return models

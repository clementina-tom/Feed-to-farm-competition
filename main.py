import yaml
import logging
import argparse
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.predictor import ModelPredictor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger("PipelineRunner")
    
    parser = argparse.ArgumentParser(description="Feed-to-Farm ML Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    try:
        # Step 1: Data Ingestion
        logger.info("--- STEP 1: DATA INGESTION ---")
        loader = DataLoader(config)
        train, test, customer, sku = loader.load_all()
        
        # Step 2: Feature Engineering
        logger.info("--- STEP 2: FEATURE ENGINEERING ---")
        engineer = FeatureEngineer()
        train, test, feature_cols = engineer.engineer_features(train, test)
        train, test, cat_cols = engineer.preprocess_metadata(train, test, customer, sku, feature_cols)
        train = engineer.generate_targets(train)
        
        # Combine numerical and categorical features — this is what models train AND predict on
        all_features = feature_cols + cat_cols
        
        # Step 3: Model Training
        logger.info("--- STEP 3: MODEL TRAINING ---")
        trainer = ModelTrainer(config)
        models = trainer.train_hybrid_ensemble(train, all_features, cat_cols)
        
        # Step 4: Prediction & Post-Processing
        logger.info("--- STEP 4: INFERENCE & POST-PROCESSING ---")
        predictor = ModelPredictor(config)
        submission = predictor.predict(models, test, all_features)
        
        logger.info(f"Pipeline executed successfully! Submission shape: {submission.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

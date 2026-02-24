# Data Directory

This directory is strictly ignored by version control (Git) to prevent exposing raw data files and large checkpoints.

### Required Datasets for Pipeline Execution

To run the `main.py` pipeline, ensure the following CSV files are downloaded from the Zindi competition portal and placed directly in the repository's root folder (or inside this directory, as long as you update `config/config.yaml` to point to them):

- `Train.csv`: Primary historical transaction dataset
- `Test.csv`: Evaluation targets
- `customer_data.csv`: Client metadata
- `sku_data.csv`: Product variant information

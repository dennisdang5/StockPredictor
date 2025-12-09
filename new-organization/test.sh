#!/bin/bash
source /Users/loganyamamoto/Desktop/class/CSCI/566/project/StockPredictor/.venv/bin/activate

# Declare model configs from main.py and create mapping file entries
# This uses the exact same configs that would be used when running the training scripts

python << 'EOF'
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import get_model_config_by_name
import util

# Get the exact configs that would be used by the training scripts
config_names = ["aelstm_base", "caelstm_base"]

print("Creating model configs for mapping file...")
print("=" * 80)

for config_name in config_names:
    try:
        # Get the exact ModelTrainingConfig from main.py
        training_config = get_model_config_by_name(config_name)
        
        # Extract the model_config (the actual model config object)
        model_config = training_config.model_config
        
        # Generate model ID and save to mapping file
        model_id = util._get_model_id(model_config)
        util._save_model_mapping(model_id, model_config)
        
        print(f"{config_name}:")
        print(f"  Model ID: {model_id}")
        print(f"  Model Type: {training_config.model_type}")
        print(f"  Config: {model_config.__class__.__name__}")
        print()
        
    except Exception as e:
        print(f"Error processing {config_name}: {e}")
        import traceback
        traceback.print_exc()

print("=" * 80)
print(f"Mapping file updated: {util.MODELS_DIR}/_model_mapping.json")
EOF
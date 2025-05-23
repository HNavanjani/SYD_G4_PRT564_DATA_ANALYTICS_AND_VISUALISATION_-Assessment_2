import os

print("Starting full pipeline...\n")

# Run existing regression pipeline
os.system("python scripts/data_cleaning.py")
os.system("python scripts/eda_analysis.py")
os.system("python scripts/preprocessing.py")
os.system("python scripts/model_training.py")
os.system("python scripts/evaluation.py")

# Run new classification pipeline
print("\n--- Starting classification model training ---")
os.system("python model/classification/classification_model.py")

print("\n--- Running classification evaluation ---")
os.system("python evaluation/classification_evaluation.py")

print("\nAll steps complete.")

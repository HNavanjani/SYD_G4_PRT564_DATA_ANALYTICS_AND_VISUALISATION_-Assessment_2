import os

print("Starting full pipeline...\n")
os.system("python scripts/data_cleaning.py")
os.system("python scripts/eda_analysis.py")
os.system("python scripts/preprocessing.py")
os.system("python scripts/model_training.py")
os.system("python scripts/evaluation.py")
print("\nAll steps complete.")

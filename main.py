"""
Main pipeline script for the Student Performance Prediction Project.

Runs:
1. Data Loading
2. Preprocessing
3. Model Training
4. Model Saving
"""

from src import preprocess_pipeline, train_model
from src.utils import print_metrics, ensure_directories


def main():
    print("\n========================================")
    print("   STUDENT PERFORMANCE PREDICTION")
    print("========================================")

    # Make sure required folders exist
    ensure_directories()

    # Path to dataset
    data_path = "data/performance.csv"

    print(f"\nLoading & preprocessing dataset: {data_path}")
    
    # Preprocess
    df = preprocess_pipeline(data_path)

    print("Preprocessing complete.")
    print("Training model...")

    # Train model
    model, auc = train_model(df)

    print("Model training complete.")
    print_metrics(auc)

    print("Model saved to: models/logistic_regression_model")
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()

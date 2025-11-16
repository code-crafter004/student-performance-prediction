from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame

import os


def train_model(df: DataFrame, model_save_path="models/logistic_regression_model"):
    """
    Train logistic regression using preprocessed data.
    """
    # Train/test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol="features", labelCol="label")

    model = lr.fit(train_df)

    # Evaluate
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    auc = evaluator.evaluate(predictions)

    print(f"Model AUC: {auc:.4f}")

    # Save model
    if not os.path.exists(model_save_path):
        model.save(model_save_path)
    else:
        print("⚠️ Model path already exists — delete or rename folder if needed.")

    return model, auc

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler


def load_model(model_path="models/logistic_regression_model"):
    """
    Load the saved PySpark model.
    """
    model = LogisticRegressionModel.load(model_path)
    return model


def prepare_single_input(attendance, marks, engagement_score):
    """
    Convert user input into Spark DataFrame + feature vector.
    """
    spark = SparkSession.builder.appName("ModelInference").getOrCreate()

    data = [(attendance, marks, engagement_score)]
    columns = ["attendance", "marks", "engagement_score"]

    df = spark.createDataFrame(data, columns)

    assembler = VectorAssembler(
        inputCols=columns,
        outputCol="features"
    )
    df = assembler.transform(df)

    return df


def predict_single(model, attendance, marks, engagement_score):
    """
    Predict pass probability for a single student.
    """
    df = prepare_single_input(attendance, marks, engagement_score)
    prediction = model.transform(df).collect()[0]

    prob = prediction.probability[1]  # Probability of passing (label=1)
    label = prediction.prediction

    return float(prob), int(label)

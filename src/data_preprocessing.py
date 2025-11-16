from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


def create_spark_session():
    return (
        SparkSession.builder
        .appName("StudentPerformancePrediction")
        .getOrCreate()
    )


# ----------------------------------------------------------------------
# STEP 1 — Load CSV
# ----------------------------------------------------------------------
def load_data(data_path):
    spark = create_spark_session()
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    return df


# ----------------------------------------------------------------------
# STEP 2 — Clean and transform dataset
# ----------------------------------------------------------------------
def clean_dataset(df):

    # Clean Homework_Completion_% column (remove "%" and convert to float)
    if "Homework_Completion_%" in df.columns:
        df = df.withColumn(
            "Homework_Completion_%", 
            regexp_replace(col("Homework_Completion_%"), "%", "").cast("double")
        )

    # Create target column final_result (because dataset does NOT contain it)
    df = df.withColumn(
        "final_result",
        when(col("Exam_Score") >= 50, "Pass").otherwise("Fail")
    )

    return df


# ----------------------------------------------------------------------
# STEP 3 — Encode label (Pass/Fail)
# ----------------------------------------------------------------------
def encode_target(df):
    indexer = StringIndexer(
        inputCol="final_result",
        outputCol="label",
        handleInvalid="keep"
    )
    df = indexer.fit(df).transform(df)
    return df


# ----------------------------------------------------------------------
# STEP 4 — Feature engineering for MLlib
# ----------------------------------------------------------------------
def engineer_features(df):

    feature_cols = []

    # numeric columns available in your dataset
    if "Exam_Score" in df.columns:
        feature_cols.append("Exam_Score")
    if "Homework_Completion_%" in df.columns:
        feature_cols.append("Homework_Completion_%")

    # Convert categorical Subject into category index
    if "Subject" in df.columns:
        subject_indexer = StringIndexer(
            inputCol="Subject",
            outputCol="Subject_index",
            handleInvalid="keep"
        )
        df = subject_indexer.fit(df).transform(df)
        feature_cols.append("Subject_index")

    # VectorAssembler for MLlib
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    df = assembler.transform(df)

    return df


# ----------------------------------------------------------------------
# PIPELINE: Full preprocessing pipeline
# ----------------------------------------------------------------------
def preprocess_pipeline(data_path):

    # Load CSV
    df = load_data(data_path)

    # Clean, add target, fix columns
    df = clean_dataset(df)

    # Encode Pass/Fail → numeric label
    df = encode_target(df)

    # Create features column
    df = engineer_features(df)

    # Drop rows with null features
    df = df.dropna(subset=["features", "label"])

    return df

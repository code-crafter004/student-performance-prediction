from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler

def load_data(path: str):
    """
    Load dataset from CSV.
    """
    spark = SparkSession.builder.appName("StudentPerformance").getOrCreate()
    
    df = spark.read.csv(path, header=True, inferSchema=True)
    return df


def clean_data(df):
    """
    Handle missing values, standardize column names.
    """
    # Drop rows with NA (optional — you can also impute)
    df = df.dropna()

    # Convert column names to consistent format (lowercase)
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.lower().replace(" ", "_"))
    
    return df


def encode_target(df):
    """
    Convert Pass/Fail → numeric labels using StringIndexer.
    """
    indexer = StringIndexer(inputCol="final_result", outputCol="label")
    df = indexer.fit(df).transform(df)
    return df


def assemble_features(df):
    """
    Combine attendance, marks, and engagement_score into one features vector.
    """
    feature_cols = ["attendance", "marks", "engagement_score"]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df)
    return df


def preprocess_pipeline(path: str):
    """
    Full preprocessing pipeline used by main.py
    """
    df = load_data(path)
    df = clean_data(df)
    df = encode_target(df)
    df = assemble_features(df)

    return df

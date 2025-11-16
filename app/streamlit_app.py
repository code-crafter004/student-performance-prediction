import streamlit as st
import pandas as pd
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

# -----------------------------
# Load PySpark Model
# -----------------------------
@st.cache_resource
def load_model():
    return LogisticRegressionModel.load("models/logistic_regression_model")


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("StreamlitSparkApp")
        .getOrCreate()
    )


# -----------------------------
# Prepare input for prediction
# -----------------------------
def prepare_features(df, spark):
    assembler = VectorAssembler(
        inputCols=["attendance", "marks", "engagement_score"],
        outputCol="features",
    )
    sdf = spark.createDataFrame(df)
    return assembler.transform(sdf)


# -----------------------------
# Risk meter function
# -----------------------------
def risk_level(prob):
    if prob >= 0.75:
        return "🟢 Safe"
    elif prob >= 0.50:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("🎓 Student Performance Prediction App")
    st.write("Upload student data and predict pass probability using PySpark ML.")

    model = load_model()
    spark = get_spark()

    # ---------------------------------------
    # Upload CSV Section
    # ---------------------------------------
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.dataframe(df)

        required_cols = {"attendance", "marks", "engagement_score"}

        if not required_cols.issubset(df.columns):
            st.error(
                f"CSV must contain columns: {required_cols}"
            )
            return

        # Convert to Spark DF with features
        spark_df = prepare_features(df, spark)

        # Predict
        predictions = model.transform(spark_df).toPandas()
        predictions["pass_probability"] = predictions["probability"].apply(lambda x: float(x[1]))
        predictions["risk"] = predictions["pass_probability"].apply(risk_level)

        st.subheader("🔮 Prediction Results")
        st.dataframe(predictions[["attendance", "marks", "engagement_score", "pass_probability", "risk"]])

    st.markdown("---")
    st.write("Created with ❤️ using PySpark + Streamlit")


if __name__ == "__main__":
    main()

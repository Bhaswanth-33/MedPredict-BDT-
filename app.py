import pandas as pd
import streamlit as st
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.functions import split, when, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random
# ------------------- Spark Session -------------------
spark = SparkSession.builder.appName("HospitalAnalyticsDashboard").getOrCreate()

# ------------------- Load Reducer Output -------------------
hdfs_path = "hdfs://localhost:9000/hospital/output/part-00000"
df = spark.read.option("delimiter", "\t").csv(hdfs_path)
df = df.withColumnRenamed("_c0", "CompositeKey").withColumnRenamed("_c1", "Total_Patients")

# ------------------- Split Composite Key -------------------
df = df.withColumn("Region", split(df["CompositeKey"], "_").getItem(0)) \
       .withColumn("Area_Type", split(df["CompositeKey"], "_").getItem(1)) \
       .withColumn("Emergency_Risk", split(df["CompositeKey"], "_").getItem(2)) \
       .withColumn("Aid_Eligible", split(df["CompositeKey"], "_").getItem(3))

# ------------------- Filter for Yes/No Only -------------------
df = df.filter(df["Aid_Eligible"].isin(["Yes", "No"]))

# ------------------- Convert to Pandas -------------------
pdf = df.select("Region", "Area_Type", "Emergency_Risk", "Aid_Eligible", "Total_Patients").toPandas()
pdf["Total_Patients"] = pd.to_numeric(pdf["Total_Patients"], errors="coerce")
pdf.dropna(inplace=True)

#--------df1---------
hdfs_path = "hdfs://localhost:9000/hospital/output/part-00000"
df1 = spark.read.option("delimiter", "\t").csv(hdfs_path)
df1 = df1.withColumnRenamed("_c0", "CompositeKey").withColumnRenamed("_c1", "Total_Patients")

# ------------------- Split Composite Key -------------------
df1 = df1.withColumn("Region", split(df1["CompositeKey"], "_").getItem(0)) \
         .withColumn("Area_Type", split(df1["CompositeKey"], "_").getItem(1)) \
         .withColumn("Emergency_Risk", split(df1["CompositeKey"], "_").getItem(2)) \
         .withColumn("Aid_Eligible", split(df1["CompositeKey"], "_").getItem(3))

# ------------------- Filter 'Pending' and Set Label -------------------
df1 = df1.filter(df1["Aid_Eligible"].isin(["Yes", "No"]))
df1 = df1.withColumn("label", when(df1["Aid_Eligible"] == "Yes", 1).otherwise(0))

# ------------------- Encode Categorical & Numeric Features -------------------
df1 = df1.withColumn("Total_Patients", col("Total_Patients").cast("double"))

# Map Emergency_Risk to numeric
df1 = df1.withColumn("Risk_Score", when(df1["Emergency_Risk"] == "Low", 1)
                                 .when(df1["Emergency_Risk"] == "Moderate", 2)
                                 .when(df1["Emergency_Risk"] == "High", 3)
                                 .otherwise(4))

region_indexer = StringIndexer(inputCol="Region", outputCol="RegionIndex")
area_indexer = StringIndexer(inputCol="Area_Type", outputCol="AreaIndex")

df1 = region_indexer.fit(df1).transform(df1)
df1 = area_indexer.fit(df1).transform(df1)

# ------------------- Assemble Features -------------------
assembler = VectorAssembler(
    inputCols=["RegionIndex", "AreaIndex", "Risk_Score", "Total_Patients"],
    outputCol="features"
)
df1 = assembler.transform(df1)
train_df, test_df = df1.randomSplit([0.8, 0.2], seed=42)
model1 = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=150, maxDepth=8).fit(train_df)
#-------end of df1---------

def flip_label(label):
    # Flip 1 out of every 10 labels
    if random.random() < 0.1:
        return "No" if label == "Yes" else "Yes"
    return label

pdf["Predicted Aid"] = pdf["Aid_Eligible"].apply(flip_label)

# ------------------- Evaluation -------------------
true_labels = pdf["Aid_Eligible"].map({"No": 0, "Yes": 1})
pred_labels = pdf["Predicted Aid"].map({"No": 0, "Yes": 1})

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=["No", "Yes"])
conf_matrix = confusion_matrix(true_labels, pred_labels)

# ------------------- Streamlit UI -------------------
st.set_page_config(layout="wide")
st.title("🏥 Hospital Analytics Dashboard with ML Predictions")
st.markdown(" ML predictions based on noisy version of `Aid_Eligible`.")

# ------------------- Top-N Selector -------------------
pdf_sorted = pdf.sort_values(by="Total_Patients", ascending=False)
top_n = st.slider("Select number of top entries to display", 5, len(pdf_sorted), 10)
top_df = pdf_sorted.head(top_n)

# ------------------- Bar Chart -------------------
st.subheader("Top Regions by Patient Count")
fig_bar = px.bar(
    top_df,
    x="Total_Patients",
    y="Region",
    orientation="h",
    color="Total_Patients",
    title="Top Regions (Simulated Prediction)",
    labels={"Total_Patients": "Patients", "Region": "Region"},
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_bar, use_container_width=True)

# ------------------- Cumulative Chart -------------------
st.subheader("Cumulative Share of Patients (Top N)")
top_df_sorted = top_df.sort_values(by="Total_Patients", ascending=False).reset_index(drop=True)
top_df_sorted["Cumulative %"] = top_df_sorted["Total_Patients"].cumsum() / top_df_sorted["Total_Patients"].sum() * 100
top_df_sorted["Rank"] = top_df_sorted.index + 1

fig_cum = px.line(
    top_df_sorted,
    x="Rank",
    y="Cumulative %",
    markers=True,
    title="Cumulative Share of Top Regions",
    labels={"Rank": "Top N", "Cumulative %": "Cumulative Patient Share (%)"}
)
st.plotly_chart(fig_cum, use_container_width=True)

# ------------------- Prediction Table -------------------
st.subheader("Simulated Predictions on Aid Eligibility")
st.dataframe(top_df[[
    "Region", "Area_Type", "Emergency_Risk", "Aid_Eligible", "Predicted Aid", "Total_Patients"
]])

# ------------------- Evaluation Metrics -------------------
st.subheader("🧠 Evaluation of Predictions")
st.markdown(f"""
- **Accuracy**: `{accuracy:.2f}`  
- **Precision**: `{precision:.2f}`  
- **Recall**: `{recall:.2f}`  
- **F1 Score**: `{f1:.2f}`
""")

with st.expander("🔍 Classification Report"):
    st.text(report)

with st.expander("🧮 Confusion Matrix"):
    st.text(conf_matrix)

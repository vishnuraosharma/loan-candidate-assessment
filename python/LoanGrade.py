import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, log1p, when
import shap
import xgboost as xgb


# Define Spark Session
spark = SparkSession.builder.appName("Loan Grader").getOrCreate()

# Path to train data
trainDataPath = "/databricks-datasets/adult/adult.data"
testDataPath = "/databricks-datasets/adult/adult.test"

# Load data
rawData = spark.read.option("header", "true").option("inferSchema", "true").csv(trainDataPath)
testRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(testDataPath)

# Preprocessing
rawData = rawData.drop("id", "loan_status")
testRaw = testRaw.drop("id")

# Define StringIndexers for categorical columns
homeOwnershipIndexer = StringIndexer(inputCol="person_home_ownership", outputCol="homeOwnershipIndex").setStringOrderType("alphabetAsc").fit(rawData)
loanIntentIndexer = StringIndexer(inputCol="loan_intent", outputCol="loanIntentIndex").setStringOrderType("alphabetAsc").fit(rawData)

# Define OneHotEncoders for indexed columns
homeOwnershipEncoder = OneHotEncoder(inputCol="homeOwnershipIndex", outputCol="homeOwnershipVec")
loanIntentEncoder = OneHotEncoder(inputCol="loanIntentIndex", outputCol="loanIntentVec")

pipeline = Pipeline(stages=[homeOwnershipIndexer, homeOwnershipEncoder, loanIntentIndexer, loanIntentEncoder])
model = pipeline.fit(rawData)

tData = model.transform(rawData)
ttData = model.transform(testRaw)

enhancedData = tData.withColumn("age_minus_history", log1p(col("person_age") - col("cb_person_cred_hist_length")))\\
                   .withColumn("actual_loan_interest", col("loan_amnt") * col("loan_int_rate"))\\
                   .withColumn("employment_stability", when(col("person_emp_length")<2, 1).otherwise(0))

enhancedTestData = ttData.withColumn("age_minus_history", log1p(col("person_age") - col("cb_person_cred_hist_length")))\\
                         .withColumn("actual_loan_interest", col("loan_amnt") * col("loan_int_rate"))\\
                         .withColumn("employment_stability", when(col("person_emp_length")<2, 1).otherwise(0))

# VectorAssembler
newassembler = VectorAssembler(
    inputCols=["homeOwnershipVec", "loanIntentVec", "defaultOnFileVec", "person_age", "person_income", "person_emp_length",
               "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "age_minus_history",
               "actual_loan_interest", "employment_stability"],
    outputCol="features")

finalOutput = newassembler.transform(enhancedData)
finalTestOutput = newassembler.transform(enhancedTestData)

# Split dataset
(trainingData, validationData) = finalOutput.randomSplit([0.8, 0.2])

# Decision tree classifier
decisionTree = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
model = decisionTree.fit(trainingData)

# Evaluators
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                              metricName="accuracy")

predictions = model.transform(validationData)
accuracy = evaluator.evaluate(predictions)

print("Test Error: ", (1.0 - accuracy))
print("Accuracy: ", accuracy)

# Convert spark dataframe to pandas dataframe for SHAP
x_test_pd = finalTestOutput.select("features").toPandas()

# Construct xgboost.DMatrix from pandas dataframe
dtest = xgb.DMatrix(x_test_pd.values)

# Train XGBoost model
model = xgb.train({"learning_rate": 0.01}, dtest, 300)

# Explain model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test_pd)

# Visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0, :], x_test_pd.iloc[0, :], matplotlib=True)
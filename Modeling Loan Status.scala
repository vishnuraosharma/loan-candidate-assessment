// Databricks notebook source
// MAGIC %md
// MAGIC # Modeling Loan Status
// MAGIC Author: Vishnu Rao-Sharma
// MAGIC
// MAGIC In this notebook we will use Kaggle's Loan Approval dataset to classify rows as approved or rejected. We perform feature engineering for the binary classification problem then use the Dense Random Forest algorithm to predict values.
// MAGIC
// MAGIC **This model will be used in an application that simultaneously predicts Loan Grade, so we will not be using the Loan Grade col as a feature**

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

// COMMAND ----------

// MAGIC %md
// MAGIC ## Preprocessing

// COMMAND ----------

val fileLocation = "/FileStore/tables/loan_train.csv"
val fileType = "csv"

// CSV options
val inferSchema = "true"
val firstRowIsHeader = "true"
val delimiter = ","

val trainWithLoanGrade = spark.read.format(fileType)
  .option("inferSchema", inferSchema)
  .option("header", firstRowIsHeader)
  .option("sep", delimiter)
  .load(fileLocation)

trainWithLoanGrade.show()  

// COMMAND ----------

val fileLocation = "dbfs:/FileStore/loan_test.csv"
val fileType = "csv"

// CSV options
val inferSchema = "true"
val firstRowIsHeader = "true"
val delimiter = ","

val testWithLoanGrade = spark.read.format(fileType)
  .option("inferSchema", inferSchema)
  .option("header", firstRowIsHeader)
  .option("sep", delimiter)
  .load(fileLocation)


// COMMAND ----------

// MAGIC %md
// MAGIC Let's drop `loan_grade`

// COMMAND ----------

val train = trainWithLoanGrade.drop("loan_grade")

val test = testWithLoanGrade.drop("loan_grade")

// COMMAND ----------

train.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Feature Engineering

// COMMAND ----------

// MAGIC %md
// MAGIC Transforming the `cb_person_default_on_file` column into binary values is essential for machine learning models, which typically require numerical input. By converting Y (Yes) to 1 and N (No) to 0, we ensure that this categorical data is represented in a format suitable for the model to process and interpret.

// COMMAND ----------

import org.apache.spark.sql.functions._

val trainTransformed = train
  .withColumn("cb_person_default_on_file_binary", when(col("cb_person_default_on_file") === "Y", 1).otherwise(0))
  .drop("cb_person_default_on_file")

trainTransformed.show()

// COMMAND ----------

// MAGIC %md
// MAGIC The `remaining_income_to_receive` column estimates the potential future income of a person, assuming they will earn their current income until the age of 80. This derived feature helps capture a proxy for a person’s long-term earning potential, which can be useful for assessing their ability to repay loans or bear financial obligations.

// COMMAND ----------

// Create the remaining_income_to_receive column
val trainIncomeLeft = trainTransformed
  .withColumn("remaining_income_to_receive", 
    col("person_income") * (lit(80) - col("person_age"))
  )

trainIncomeLeft.show()

// COMMAND ----------

// MAGIC %md
// MAGIC The `remaining_employment_length` column estimates the number of working years left for a person, assuming retirement at age 65. This feature can provide insights into the individual's earning potential and financial stability, particularly for long-term loans.

// COMMAND ----------

// Create the remaining_employment_length column
val trainRetirement = trainIncomeLeft
  .withColumn("remaining_employment_length", 
    lit(65) - col("person_age")
  )

trainRetirement.show()

// COMMAND ----------

// MAGIC %md
// MAGIC The `age_to_history_length_ratio` column calculates the ratio of a person's age to their credit history length. This feature provides insights into how long the individual has been building their credit relative to their age. A lower ratio might indicate a well-established credit history for their age, which could be a positive indicator of creditworthiness.

// COMMAND ----------

// Create the age_to_history_length_ratio column
val trainAgetoHist = trainRetirement
  .withColumn("age_to_history_length_ratio", 
    col("person_age") / col("cb_person_cred_hist_length")
  )

trainAgetoHist.show()

// COMMAND ----------

// MAGIC %md
// MAGIC Income, being a continuous variable, can introduce noise in machine learning models if there are extreme outliers or a non-linear relationship with the target variable. By grouping income into brackets, we can simplify the feature, reduce noise, and potentially capture non-linear effects. The provided UDF creates six income brackets based on specified thresholds, which will replace the original continuous values

// COMMAND ----------

import org.apache.spark.sql.expressions.UserDefinedFunction
// Define the UDF for income brackets
val incomeBuckets: UserDefinedFunction = udf((income: Int) => income match {
  case i if i < 30000 => 1
  case i if i >= 30000 && i < 50000 => 2
  case i if i >= 50000 && i < 70000 => 3
  case i if i >= 70000 && i < 100000 => 4
  case i if i >= 100000 && i < 150000 => 5
  case _ => 6
})

// Apply the UDF to create a new column for income brackets
val trainBracketed = trainAgetoHist
  .withColumn("income_bracket", incomeBuckets(col("person_income")))
  .drop("person_income") // Drop the original column

trainBracketed.show()


// COMMAND ----------

// MAGIC %md
// MAGIC The `credit_risk` column calculates a proxy for the risk associated with extending credit to a person. By combining `loan_amnt`, `loan_int_rate`, and `cb_person_cred_hist_length`, this metric accounts for the size of the loan, the interest rate (indicating perceived risk), and the person's credit history length (a proxy for creditworthiness). This derived feature can be valuable for predicting loan default probability.

// COMMAND ----------

// Create the credit risk column
val trainCreditRisk = trainBracketed
  .withColumn("credit_risk", 
    (col("loan_amnt") * col("loan_int_rate")) / col("cb_person_cred_hist_length")
  )

trainCreditRisk.show()

// COMMAND ----------

// MAGIC %md
// MAGIC For the `loan_intent` column, one-hot encoding will create a new binary column for each unique value in loan_intent, where each column will have a 1 if the row corresponds to that intent and 0 otherwise.

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SparseVector

 // Step 1: Create a StringIndexer for "person_home_ownership" to convert the categorical column into numeric indices
  val indexer1 = new StringIndexer()
    .setInputCol("person_home_ownership")
    .setOutputCol("person_home_ownership_index")

  // Step 2: Use OneHotEncoder to one-hot encode the indexed column
  val encoder1 = new OneHotEncoder()
    .setInputCol("person_home_ownership_index")
    .setOutputCol("person_home_ownership_onehot")

  // Step 3: Create a pipeline for the "person_home_ownership" transformations
  val pipeline1 = new Pipeline().setStages(Array(indexer1, encoder1))

  // Step 4: Fit and transform the pipeline on the training DataFrame
  val model1 = pipeline1.fit(trainCreditRisk)
  val encodedDF1 = model1.transform(trainCreditRisk)

  // Step 5: Convert the one-hot encoded column to separate columns
  val extractOneHot = udf((vec: SparseVector) => {
    val numCategories = vec.size
    (0 until numCategories).map(i => if (vec(i) == 1.0) 1 else 0).toArray
  })

  val homefinalDF = encodedDF1.withColumn("onehot_split", extractOneHot(col("person_home_ownership_onehot")))

  // Step 6: Create a sequence of encoded column names for the one-hot values
  val oneHotColumns1 = (0 until encodedDF1.select("person_home_ownership_onehot").first().getAs[SparseVector]("person_home_ownership_onehot").size)
    .map(i => s"person_home_ownership_onehot_$i")

  // Step 7: Add the one-hot encoded columns dynamically for "person_home_ownership"
  val trainHome = oneHotColumns1.zipWithIndex.foldLeft(homefinalDF) { case (df, (colName, index)) =>
    df.withColumn(colName, col("onehot_split").getItem(index))
      .drop("person_home_ownership")
      .drop("person_home_ownership_index")
  }

trainHome.show()


// COMMAND ----------

// MAGIC %md
// MAGIC And again for loan_intent

// COMMAND ----------

// Step 8: Repeat the same process for "loan_intent" column
  val indexer2 = new StringIndexer()
    .setInputCol("loan_intent")
    .setOutputCol("loan_intent_index")

  val encoder2 = new OneHotEncoder()
    .setInputCol("loan_intent_index")
    .setOutputCol("loan_intent_onehot")

  val pipeline2 = new Pipeline().setStages(Array(indexer2, encoder2))

  val model2 = pipeline2.fit(trainHome)
  val encodedDF2 = model2.transform(trainHome)

  val intentfinalDF = encodedDF2.withColumn("onehot_split", extractOneHot(col("loan_intent_onehot")))

  // Step 9: Create a sequence of encoded column names for the one-hot values of "loan_intent"
  val oneHotColumns2 = (0 until encodedDF2.select("loan_intent_onehot").first().getAs[SparseVector]("loan_intent_onehot").size)
    .map(i => s"loan_intent_onehot_$i")

  // Step 10: Add the one-hot encoded columns dynamically for "loan_intent"
  val finalTrain = oneHotColumns2.zipWithIndex.foldLeft(intentfinalDF) { case (df, (colName, index)) =>
    df.withColumn(colName, col("onehot_split").getItem(index))
      .drop("loan_intent")
      .drop("loan_intent_index")
      .drop("person_home_ownership_index")
  }

// COMMAND ----------

finalTrain.columns

// COMMAND ----------

// MAGIC %md
// MAGIC ### The Pipeline
// MAGIC Let's create a function with all the transformations and apply them to the test set.

// COMMAND ----------


def transformData(df: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.Dataset[_]  =  {
  val testTransformed = df
    .withColumn("cb_person_default_on_file_binary", when(col("cb_person_default_on_file") === "Y", 1).otherwise(0))
    .drop("cb_person_default_on_file")

  // Create the remaining_income_to_receive column
  val testIncomeLeft = testTransformed
    .withColumn("remaining_income_to_receive", 
      col("person_income") * (lit(80) - col("person_age"))
    )

    // Create the remaining_employment_length column
  val testRetirement = testIncomeLeft
    .withColumn("remaining_employment_length", 
      lit(65) - col("person_age")
    )

  val testAgetoHist = testRetirement
    .withColumn("age_to_history_length_ratio", 
      col("person_age") / col("cb_person_cred_hist_length")
    )
  // Define the UDF for income brackets
  val incomeBuckets: UserDefinedFunction = udf((income: Int) => income match {
    case i if i < 30000 => 1
    case i if i >= 30000 && i < 50000 => 2
    case i if i >= 50000 && i < 70000 => 3
    case i if i >= 70000 && i < 100000 => 4
    case i if i >= 100000 && i < 150000 => 5
    case _ => 6
  })

  // Apply the UDF to create a new column for income brackets
  val testBracketed = testAgetoHist
    .withColumn("income_bracket", incomeBuckets(col("person_income")))
    .drop("person_income") // Drop the original column

  val testCreditRisk = testBracketed
    .withColumn("credit_risk", 
      (col("loan_amnt") * col("loan_int_rate")) / col("cb_person_cred_hist_length")
    )

  // Step 1: Create a StringIndexer for "person_home_ownership" to convert the categorical column into numeric indices
  val indexer1 = new StringIndexer()
    .setInputCol("person_home_ownership")
    .setOutputCol("person_home_ownership_index")

  // Step 2: Use OneHotEncoder to one-hot encode the indexed column
  val encoder1 = new OneHotEncoder()
    .setInputCol("person_home_ownership_index")
    .setOutputCol("person_home_ownership_onehot")

  // Step 3: Create a pipeline for the "person_home_ownership" transformations
  val pipeline1 = new Pipeline().setStages(Array(indexer1, encoder1))

  // Step 4: Fit and transform the pipeline on the training DataFrame
  val model1 = pipeline1.fit(testCreditRisk)
  val encodedDF1 = model1.transform(testCreditRisk)

  // Step 5: Convert the one-hot encoded column to separate columns
  val extractOneHot = udf((vec: SparseVector) => {
    val numCategories = vec.size
    (0 until numCategories).map(i => if (vec(i) == 1.0) 1 else 0).toArray
  })

  val homefinalDF = encodedDF1.withColumn("onehot_split", extractOneHot(col("person_home_ownership_onehot")))

  // Step 6: Create a sequence of encoded column names for the one-hot values
  val oneHotColumns1 = (0 until encodedDF1.select("person_home_ownership_onehot").first().getAs[SparseVector]("person_home_ownership_onehot").size)
    .map(i => s"person_home_ownership_onehot_$i")

  // Step 7: Add the one-hot encoded columns dynamically for "person_home_ownership"
  val testHome = oneHotColumns1.zipWithIndex.foldLeft(homefinalDF) { case (df, (colName, index)) =>
    df.withColumn(colName, col("onehot_split").getItem(index))
      .drop("person_home_ownership")
  }

  // Step 8: Repeat the same process for "loan_intent" column
  val indexer2 = new StringIndexer()
    .setInputCol("loan_intent")
    .setOutputCol("loan_intent_index")

  val encoder2 = new OneHotEncoder()
    .setInputCol("loan_intent_index")
    .setOutputCol("loan_intent_onehot")

  val pipeline2 = new Pipeline().setStages(Array(indexer2, encoder2))

  val model2 = pipeline2.fit(testHome)
  val encodedDF2 = model2.transform(testHome)

  val intentfinalDF = encodedDF2.withColumn("onehot_split", extractOneHot(col("loan_intent_onehot")))

  // Step 9: Create a sequence of encoded column names for the one-hot values of "loan_intent"
  val oneHotColumns2 = (0 until encodedDF2.select("loan_intent_onehot").first().getAs[SparseVector]("loan_intent_onehot").size)
    .map(i => s"loan_intent_onehot_$i")

  // Step 10: Add the one-hot encoded columns dynamically for "loan_intent"
  val final_df = oneHotColumns2.zipWithIndex.foldLeft(intentfinalDF) { case (df, (colName, index)) =>
    df.withColumn(colName, col("onehot_split").getItem(index))
      .drop("loan_intent")
      .drop("loan_intent_index")
      .drop("person_home_ownership_index")
  }

  // Return the final DataFrame with the one-hot encoded columns
  return final_df

}
 

// COMMAND ----------

finalTrain.select("id", "person_age", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "loan_status", "cb_person_default_on_file_binary").orderBy($"id".desc).show(5)

// COMMAND ----------

val finalTest = transformData(test)

// COMMAND ----------

finalTest.select("id", "person_age", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "cb_person_default_on_file_binary").orderBy($"id".desc).show(5)

// COMMAND ----------

finalTest.columns

// COMMAND ----------

// MAGIC %md
// MAGIC ## Modeling
// MAGIC Let's try and make some predictions for our test set with the following:
// MAGIC 1. Random Forest
// MAGIC 2. Logistic Regression
// MAGIC
// MAGIC We will submit these predictions to [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e10/code?competitionId=84894&sortBy=scoreDescending&excludeNonAccessedDatasources=true) to see guage model precision.

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ### Modeling with Random Forest
// MAGIC When modeling this feature set and making prediction for Loan Status with logistic regression, out submitted results have a precision of 0.8 on the test set. 

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col

// Specify feature columns
val featureCols = Array(
  "person_age", 
  "person_emp_length", 
  "loan_amnt", 
  "loan_int_rate", 
  "loan_percent_income", 
  "cb_person_cred_hist_length", 
  "cb_person_default_on_file_binary", 
  "remaining_income_to_receive", 
  "remaining_employment_length", 
  "age_to_history_length_ratio", 
  "income_bracket", 
  "credit_risk", 
  "person_home_ownership_onehot_0",
  "person_home_ownership_onehot_1", 
  "person_home_ownership_onehot_2",
  "loan_intent_onehot",
  "loan_intent_onehot_0", 
  "loan_intent_onehot_1", 
  "loan_intent_onehot_2", 
  "loan_intent_onehot_3", 
  "loan_intent_onehot_4"
)

// Create a VectorAssembler to combine feature columns into a single vector column
val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

// Initialize RandomForestClassifier
val rf = new RandomForestClassifier()
  .setLabelCol("loan_status")
  .setFeaturesCol("features")
  .setNumTrees(100)  
  .setMaxDepth(10)   
  .setMinInstancesPerNode(5)  

// Create a Pipeline to chain the assembler and the classifier
val pipeline = new Pipeline().setStages(Array(assembler, rf))

// Train the model & make predictions
val model = pipeline.fit(finalTrain)
val predictions = model.transform(finalTest)

val output = predictions.select(
  col("id"),  
  col("prediction").cast("int").as("loan_status_prediction")
)


// Saving the model, Print out the feature importances
val featureImportances = model.featureImportances
println(s"Feature Importances: $featureImportances")
model
  .write
  .overwrite()
  .save("./downloads")


// COMMAND ----------

display(output)


// COMMAND ----------

// MAGIC %md
// MAGIC ### Modeling with Logistic Regression
// MAGIC When modeling this feature set and making prediction for Loan Status with logistic regression, out submitted results have a precision of 0.69 on the test set. 

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col

// Specify feature columns
val featureCols = Array(
  "person_age", 
  "person_emp_length", 
  "loan_amnt", 
  "loan_int_rate", 
  "loan_percent_income", 
  "cb_person_cred_hist_length", 
  "cb_person_default_on_file_binary", 
  "remaining_income_to_receive", 
  "remaining_employment_length", 
  "age_to_history_length_ratio", 
  "income_bracket", 
  "credit_risk", 
  "person_home_ownership_onehot_0",
  "person_home_ownership_onehot_1", 
  "person_home_ownership_onehot_2",
  "loan_intent_onehot",
  "loan_intent_onehot_0", 
  "loan_intent_onehot_1", 
  "loan_intent_onehot_2", 
  "loan_intent_onehot_3", 
  "loan_intent_onehot_4"
)

// Create a VectorAssembler to combine feature columns into a single vector column
val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

// Initialize LogisticRegression
val lr = new LogisticRegression()
  .setLabelCol("loan_status")
  .setFeaturesCol("features")
  .setMaxIter(100)  // Number of iterations for optimization
  .setRegParam(0.01)  // Regularization parameter (L2 penalty)

// Create a Pipeline to chain the assembler and the classifier
val pipeline1 = new Pipeline().setStages(Array(assembler, lr))

// Train the model & make predictions
val model1 = pipeline1.fit(finalTrain)
val predictions1 = model1.transform(finalTest)

val output2 = predictions1.select(
  col("id"),  
  col("prediction").cast("int").as("loan_status_prediction")
)


// COMMAND ----------

display(output2)


// COMMAND ----------

// MAGIC %md
// MAGIC ## Model Evaluation
// MAGIC We will take a couple more measures of the more performant, Random Forest classifier to understand its overall capabilities using cross-fold validation:
// MAGIC 1. Precision
// MAGIC 2. Recall
// MAGIC
// MAGIC With 5-fold cross validation on the train set, we acheive a precision of 0.97 and a recall of 0.63. 
// MAGIC
// MAGIC Among all the samples predicted as positive (loan approved), 93% are actual positives.
// MAGIC This suggests the model has low false positives (FP). It is selective in predicting a loan approval and avoids incorrectly approving loans that should be rejected.
// MAGIC
// MAGIC The precision is 63% meaning about 37% of loans that should have been approved are missed (false negatives, FN).
// MAGIC
// MAGIC The model leans toward being conservative, prioritizing precision over recall - this is a desirable outcome for us as it protects grantors from approving poor application and gives requesters a realistic picture.

// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.col

// Create ParamGridBuilder
val paramGrid = new ParamGridBuilder()
  .build()

// Define BinaryClassificationEvaluator
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("loan_status")
  .setRawPredictionCol("rawPrediction")
  .setMetricName("areaUnderROC") 

// Set up CrossValidator with 5 folds
val crossValidator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5) 

// Perform cross-validation and get the best model
val cvModel = crossValidator.fit(finalTrain)

// Retrieve predictions from cross-validation
val predictions = cvModel.transform(finalTrain)

// Calculate precision and recall directly
val tp = predictions.filter(col("loan_status") === 1.0 && col("prediction") === 1.0).count()
val fp = predictions.filter(col("loan_status") === 0.0 && col("prediction") === 1.0).count()
val fn = predictions.filter(col("loan_status") === 1.0 && col("prediction") === 0.0).count()

val precision = tp.toDouble / (tp + fp)
val recall = tp.toDouble / (tp + fn)

println(s"Precision: $precision")
println(s"Recall: $recall")


// COMMAND ----------

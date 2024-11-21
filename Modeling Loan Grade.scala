
// Spark Session
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder
  .appName("Loan Grader")
  .config("spark.master", "local")
  .getOrCreate()

import spark.implicits._

//Data Preprocessing
val trainDataPath = "/FileStore/tables/train-2.csv"
val testDataPath = "/FileStore/tables/test-3.csv"

var rawData = spark.read.option("header", "true").option("inferSchema", "true").csv(trainDataPath)
var testRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(testDataPath)
rawData.show()

// Preprocessing
rawData = rawData.drop("id", "loan_status")
testRaw = testRaw.drop("id")
rawData.show()


// Encoding and Indexing
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.Pipeline

// Index labels, adding metadata to the label column. Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("loan_grade").setOutputCol("indexedLabel").fit(rawData)

// Index and then OneHotEncode features
val homeOwnershipIndexer = new StringIndexer().setInputCol("person_home_ownership").setOutputCol("homeOwnershipIndex").fit(rawData)
val homeOwnershipEncoder = new OneHotEncoder().setInputCol("homeOwnershipIndex").setOutputCol("homeOwnershipVec")

val loanIntentIndexer = new StringIndexer().setInputCol("loan_intent").setOutputCol("loanIntentIndex").fit(rawData)
val loanIntentEncoder = new OneHotEncoder().setInputCol("loanIntentIndex").setOutputCol("loanIntentVec")

val defaultOnFileIndexer = new StringIndexer().setInputCol("cb_person_default_on_file").setOutputCol("defaultOnFileIndex").fit(rawData)
val defaultOnFileEncoder = new OneHotEncoder().setInputCol("defaultOnFileIndex").setOutputCol("defaultOnFileVec")

// Assemble everything together and add continuous features, then output a column called "features"

val assembler = new VectorAssembler()
  .setInputCols(Array("homeOwnershipVec", "loanIntentVec", "defaultOnFileVec", "person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"))
  .setOutputCol("features")

// Pipeline: stringindexing + onehotencoding + vectorassembling stages
val pipeline = new Pipeline().setStages(Array(homeOwnershipIndexer, homeOwnershipEncoder, loanIntentIndexer, loanIntentEncoder, defaultOnFileIndexer, defaultOnFileEncoder, labelIndexer, assembler))

// Fit the pipeline to training documents.
val model = pipeline.fit(rawData)
val ttData = model.transform(testRaw)
val tData = model.transform(rawData)
val transformedData = tData.drop("person_home_ownership")
  .drop("loan_intent")
  .drop("cb_person_default_on_file")
  .drop("loan_grade")
val transformedTestData = ttData.drop("person_home_ownership")
  .drop("loan_intent")
  .drop("cb_person_default_on_file")
  .drop("loan_grade")

// Adding New Features
//transformedTestData.show()
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.log1p

val enhancedData = transformedData
  .withColumn("age_minus_history", log1p(col("person_age") - col("cb_person_cred_hist_length")))
  .withColumn("actual_loan_interest", col("loan_amnt") * col("loan_int_rate"))
  .withColumn("employment_stability", when(col("person_emp_length") < 2, 1).otherwise(0))

// Test Data set does the same
val enhancedTestData = transformedTestData
  .withColumn("age_minus_history", log1p(col("person_age") - col("cb_person_cred_hist_length")))
  .withColumn("actual_loan_interest", col("loan_amnt") * col("loan_int_rate"))
  .withColumn("employment_stability", when(col("person_emp_length") < 2, 1).otherwise(0))

enhancedTestData.show()
//enhancedData.show()

// Putting it all Together
val newassembler = new VectorAssembler()
  .setInputCols(Array("homeOwnershipVec", "loanIntentVec", "defaultOnFileVec", "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "age_minus_history",
    "actual_loan_interest", "employment_stability"))
  .setOutputCol("new_features").setHandleInvalid("skip")

val output = newassembler.transform(enhancedData)
val finalOutput = output.drop("features").withColumnRenamed("new_features", "features")
//finalOutput.show()

// Test Assembler to put it all together
val testOutput = newassembler.transform(enhancedTestData)
val finalTestOutput = testOutput.drop("features").withColumnRenamed("new_features", "features")

finalTestOutput.show()


// Splitting into Training and Validation
val Array(trainingData, validationData) = finalOutput.randomSplit(Array(0.8, 0.2))
finalOutput.columns
trainingData.printSchema()

// Model
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

// Set up the layers of the neural network
// The first layer is the input layer (number of features),
// the middle layers are the hidden layers
// the last layer is the output layer (number of classes)
val layers = Array[Int](19, 64, 32, 64, 7)  // Change 7 to the number of classes you have

val mlp = new MultilayerPerceptronClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("features")
  .setMaxIter(100)
  .setLayers(layers)

val mlpModel = mlp.fit(trainingData)
val mlpPredictions = mlpModel.transform(validationData)


// Evaluation
// Evaluate accuracy
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy = evaluator.evaluate(mlpPredictions)
println(s"Test Accuracy = $accuracy")

val testPredictions = model.transform(finalTestOutput)
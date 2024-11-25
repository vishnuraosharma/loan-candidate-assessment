
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
val labelIndexer = new StringIndexer()
  .setInputCol("loan_grade")
  .setOutputCol("indexedLabel")
  .setStringOrderType("alphabetAsc")  // This will order labels alphabetically
  .fit(rawData)

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
//val layers = Array[Int](19, 64, 32, 64, 7)  // Change 7 to the number of classes you have
//
//val mlp = new MultilayerPerceptronClassifier()
//  .setLabelCol("indexedLabel")
//  .setFeaturesCol("features")
//  .setMaxIter(100)
//  .setLayers(layers)
//
//val mlpModel = mlp.fit(trainingData)
//val mlpPredictions = mlpModel.transform(validationData)
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

// Reduce the number of layers and neurons to simplify the network
val layers = Array[Int](19, 15, 15, 7)  // Use fewer neurons

// Initialize the MLP
val mlp = new MultilayerPerceptronClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("features")
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setSolver("l-bfgs")  // Use 'sgd' solver for faster convergence
  .setTol(1E-6)
  .setStepSize(0.03)

// Define the parameter grid with fewer combinations
val paramGrid = new ParamGridBuilder()
  .addGrid(mlp.maxIter, Array(50, 100))
  .addGrid(mlp.layers, Array(Array(19, 15, 15, 7)))  // Only one configuration to try
  .build()

// Evaluator
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// Cross Validator
val cv = new CrossValidator()
  .setEstimator(mlp)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)  // Use 3 folds for faster results
  .setParallelism(1)  // Keep parallelism low for a single machine

// Run cross-validation
val cvModel = cv.fit(trainingData)

// Use the test data to measure the accuracy
val mlpPredictions = cvModel.transform(validationData)

// Evaluate the best model
val accuracy = evaluator.evaluate(mlpPredictions)
println(s"Test Accuracy = $accuracy")



// Evaluation
// Evaluate accuracy
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//val evaluator = new MulticlassClassificationEvaluator()
//  .setLabelCol("indexedLabel")
//  .setPredictionCol("prediction")
//  .setMetricName("accuracy")
//
//val accuracy = evaluator.evaluate(mlpPredictions)
//println(s"Test Accuracy = $accuracy")
//
//val testPredictions = model.transform(finalTestOutput)
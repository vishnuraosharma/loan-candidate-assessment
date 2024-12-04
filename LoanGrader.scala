import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import java.io.IOException

object LoanGrader {
  def main(args: Array[String]): Unit = {
    // Spark Session with additional configurations
    System.setProperty("hadoop.home.dir", "C:\\winutils\\")
    val spark = SparkSession.builder
      .appName("Loan Grader")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .getOrCreate()

    try {
      import spark.implicits._

      // Data Preprocessing
      val trainDataPath = ""
      val testDataPath = ""

      val rawDataInitial = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(trainDataPath)

      val testRawInitial = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(testDataPath)

      // Preprocessing - chain transformations instead of reassignment
      val rawData = rawDataInitial.drop("id", "loan_status")
      val testRaw = testRawInitial.drop("id")
      rawData.show()

      // Index labels, adding metadata to the label column
      val labelIndexer = new StringIndexer()
        .setInputCol("loan_grade")
        .setOutputCol("indexedLabel")
        .setStringOrderType("alphabetAsc")
        .fit(rawData)

      // Index and then OneHotEncode features
      val homeOwnershipIndexer = new StringIndexer().setInputCol("person_home_ownership").setOutputCol("homeOwnershipIndex").fit(rawData)
      val homeOwnershipEncoder = new OneHotEncoder().setInputCol("homeOwnershipIndex").setOutputCol("homeOwnershipVec")

      val loanIntentIndexer = new StringIndexer().setInputCol("loan_intent").setOutputCol("loanIntentIndex").fit(rawData)
      val loanIntentEncoder = new OneHotEncoder().setInputCol("loanIntentIndex").setOutputCol("loanIntentVec")

      val defaultOnFileIndexer = new StringIndexer().setInputCol("cb_person_default_on_file").setOutputCol("defaultOnFileIndex").fit(rawData)
      val defaultOnFileEncoder = new OneHotEncoder().setInputCol("defaultOnFileIndex").setOutputCol("defaultOnFileVec")

      val assembler = new VectorAssembler()
        .setInputCols(Array("homeOwnershipVec", "loanIntentVec", "defaultOnFileVec", "person_age",
          "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income",
          "cb_person_cred_hist_length"))
        .setOutputCol("features")

      // Pipeline stages
      val pipeline = new Pipeline().setStages(Array(homeOwnershipIndexer, homeOwnershipEncoder,
        loanIntentIndexer, loanIntentEncoder, defaultOnFileIndexer, defaultOnFileEncoder,
        labelIndexer, assembler))

      // Fit the pipeline
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
      val enhancedData = transformedData
        .withColumn("age_minus_history", log1p(col("person_age") - col("cb_person_cred_hist_length")))
        .withColumn("actual_loan_interest", col("loan_amnt") * col("loan_int_rate"))
        .withColumn("employment_stability", when(col("person_emp_length") < 2, 1).otherwise(0))

      val enhancedTestData = transformedTestData
        .withColumn("age_minus_history", log1p(col("person_age") - col("cb_person_cred_hist_length")))
        .withColumn("actual_loan_interest", col("loan_amnt") * col("loan_int_rate"))
        .withColumn("employment_stability", when(col("person_emp_length") < 2, 1).otherwise(0))

      // Final feature assembly
      val newassembler = new VectorAssembler()
        .setInputCols(Array("homeOwnershipVec", "loanIntentVec", "defaultOnFileVec", "person_age",
          "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income",
          "cb_person_cred_hist_length", "age_minus_history", "actual_loan_interest", "employment_stability"))
        .setOutputCol("new_features")
        .setHandleInvalid("skip")

      val output = newassembler.transform(enhancedData)
      val finalOutput = output.drop("features").withColumnRenamed("new_features", "features")

      val testOutput = newassembler.transform(enhancedTestData)
      val finalTestOutput = testOutput.drop("features").withColumnRenamed("new_features", "features")

      // Get actual feature count
      val featureCount = finalOutput.select("features").first()(0)
        .asInstanceOf[org.apache.spark.ml.linalg.Vector].size
      println(s"Actual feature count: $featureCount")

      // Split data
      val Array(trainingData, validationData) = finalOutput.randomSplit(Array(0.8, 0.2))

      // Define the neural network with dynamic feature count
      val layers = Array[Int](featureCount, 15, 15, 7)

      val mlp = new MultilayerPerceptronClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("features")
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
        .setSolver("l-bfgs")
        .setTol(1E-6)
        .setStepSize(0.03)
        .setMaxIter(100)

      // Simplified parameter grid
      val paramGrid = new ParamGridBuilder()
        .addGrid(mlp.maxIter, Array(100))
        .addGrid(mlp.stepSize, Array(0.03))
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
        .setNumFolds(3)
        .setParallelism(1)

      // Run cross-validation and get predictions
      val cvModel = cv.fit(trainingData)
      val mlpPredictions = cvModel.transform(validationData)

      // Evaluate the model
      val accuracy = evaluator.evaluate(mlpPredictions)
      println("\nModel Performance:")
      println(s"Test Accuracy = $accuracy")

      // Make predictions on test data
      val testPredictions = cvModel.transform(finalTestOutput)
      println("\nPredictions on test data:")
      testPredictions.select("prediction").show()

      // Save the model with simplified path handling
      try {
        val modelPath = ""
        cvModel.save(modelPath)  // Simplified saving approach
        println(s"Model saved to: $modelPath")
      } catch {
        case e: IOException =>
          println(s"IO Error while saving model: ${e.getMessage}")
        case e: Exception =>
          println(s"Warning: Could not save model: ${e.getMessage}")
          e.printStackTrace()
      }

    } catch {
      case e: Exception =>
        println(s"Error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}
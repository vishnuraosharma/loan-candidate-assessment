import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SparseVector

import java.io.IOException

object LoanStatus {
  def main(args: Array[String]): Unit = {
    // Initialize Spark Session
    System.setProperty("hadoop.home.dir", "C:\\winutils\\")
    val spark = SparkSession.builder
      .appName("Loan Status Predictor")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .getOrCreate()

    try {
      import spark.implicits._

      // Load Data
      val trainDataPath = ""
      val testDataPath = ""

      val trainWithLoanGrade = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(trainDataPath)

      val testWithLoanGrade = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(testDataPath)

      // Drop loan_grade column
      val train = trainWithLoanGrade.drop("loan_grade")
      val test = testWithLoanGrade.drop("loan_grade")

      // Feature Engineering Function
      def transformData(df: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.Dataset[_] = {
        // Binary conversion for default_on_file
        val dfTransformed = df
          .withColumn("cb_person_default_on_file_binary",
            when(col("cb_person_default_on_file") === "Y", 1).otherwise(0))
          .drop("cb_person_default_on_file")

        // Calculate remaining income
        val dfIncome = dfTransformed
          .withColumn("remaining_income_to_receive",
            col("person_income") * (lit(80) - col("person_age")))

        // Calculate remaining employment length
        val dfEmployment = dfIncome
          .withColumn("remaining_employment_length",
            lit(65) - col("person_age"))

        // Calculate age to history ratio
        val dfAgeRatio = dfEmployment
          .withColumn("age_to_history_length_ratio",
            col("person_age") / col("cb_person_cred_hist_length"))

        // Income brackets
        val incomeBuckets = udf((income: Int) => income match {
          case i if i < 30000 => 1
          case i if i >= 30000 && i < 50000 => 2
          case i if i >= 50000 && i < 70000 => 3
          case i if i >= 70000 && i < 100000 => 4
          case i if i >= 100000 && i < 150000 => 5
          case _ => 6
        })

        val dfBrackets = dfAgeRatio
          .withColumn("income_bracket", incomeBuckets(col("person_income")))
          .drop("person_income")

        // Calculate credit risk
        val dfRisk = dfBrackets
          .withColumn("credit_risk",
            (col("loan_amnt") * col("loan_int_rate")) / col("cb_person_cred_hist_length"))

        // One-hot encoding for categorical variables
        val homeOwnershipIndexer = new StringIndexer()
          .setInputCol("person_home_ownership")
          .setOutputCol("homeOwnershipIndex")
          .fit(dfRisk)

        val homeOwnershipEncoder = new OneHotEncoder()
          .setInputCol("homeOwnershipIndex")
          .setOutputCol("homeOwnershipVec")

        val loanIntentIndexer = new StringIndexer()
          .setInputCol("loan_intent")
          .setOutputCol("loanIntentIndex")
          .fit(dfRisk)

        val loanIntentEncoder = new OneHotEncoder()
          .setInputCol("loanIntentIndex")
          .setOutputCol("loanIntentVec")

        val pipeline = new Pipeline().setStages(Array(
          homeOwnershipIndexer, homeOwnershipEncoder,
          loanIntentIndexer, loanIntentEncoder
        ))

        pipeline.fit(dfRisk).transform(dfRisk)
      }

      // Transform train and test data
      val transformedTrain = transformData(train)
      val transformedTest = transformData(test)

      // Prepare features for model
      val featureCols = Array(
        "person_age", "person_emp_length", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_cred_hist_length",
        "cb_person_default_on_file_binary", "remaining_income_to_receive",
        "remaining_employment_length", "age_to_history_length_ratio",
        "income_bracket", "credit_risk", "homeOwnershipVec", "loanIntentVec"
      )

      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")

      // Initialize Random Forest
      val rf = new RandomForestClassifier()
        .setLabelCol("loan_status")
        .setFeaturesCol("features")
        .setNumTrees(100)
        .setMaxDepth(10)
        .setMinInstancesPerNode(5)

      // Create and fit pipeline
      val modelPipeline = new Pipeline()
        .setStages(Array(assembler, rf))

      val model = modelPipeline.fit(transformedTrain)

      // Make predictions
      val predictions = model.transform(transformedTest)

      println("\nSample Predictions:")
      predictions.select("prediction").show(5)

      // Save the model with simplified path handling
      try {
        val modelPath = ""
        model.save(modelPath)  // Simplified saving approach
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
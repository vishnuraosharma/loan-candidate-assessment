import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.ml.linalg.SparseVector

object LoanApprovalApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("LoanApprovalApp")
      .master("local[*]")
      .getOrCreate()

    // Load training data
    val trainWithLoanGrade = spark.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("sep", ",")
      .load("C:/Users/momog/Downloads/train.csv")

    // Load test data
    val testWithLoanGrade = spark.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("sep", ",")
      .load("C:/Users/momog/Downloads/test.csv")

    // Data preprocessing and feature engineering
    val train = preprocessData(trainWithLoanGrade.drop("loan_grade"))
    val test = preprocessData(testWithLoanGrade.drop("loan_grade"))

    // Train RandomForest model
    val model = trainRandomForestModel(train)

    // Save the model
    val modelPath = "C:/Users/momog/Downloads/loan_status_model"
    model.write.overwrite().save(modelPath)
    println(s"Model saved at: $modelPath")

    spark.stop()
  }

  def preprocessData(df: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    // Your preprocessing code here, similar to the notebook
    // Transformations, feature engineering, etc.
    import df.sparkSession.implicits._

    // Convert 'cb_person_default_on_file' to binary
    val dfTransformed = df
      .withColumn("cb_person_default_on_file_binary", when(col("cb_person_default_on_file") === "Y", 1).otherwise(0))
      .drop("cb_person_default_on_file")

    // Create the 'remaining_income_to_receive' column
    val dfIncomeLeft = dfTransformed
      .withColumn("remaining_income_to_receive",
        col("person_income") * (lit(80) - col("person_age"))
      )

    // Create the 'remaining_employment_length' column
    val dfRetirement = dfIncomeLeft
      .withColumn("remaining_employment_length",
        lit(65) - col("person_age")
      )

    // Create the 'age_to_history_length_ratio' column
    val dfAgetoHist = dfRetirement
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
    val dfBracketed = dfAgetoHist
      .withColumn("income_bracket", incomeBuckets(col("person_income")))
      .drop("person_income") // Drop the original column

    // Create the 'credit_risk' column
    val dfCreditRisk = dfBracketed
      .withColumn("credit_risk",
        (col("loan_amnt") * col("loan_int_rate")) / col("cb_person_cred_hist_length")
      )

    // One-hot encode 'person_home_ownership'
    val indexer1 = new StringIndexer()
      .setInputCol("person_home_ownership")
      .setOutputCol("person_home_ownership_index")

    val encoder1 = new OneHotEncoder()
      .setInputCol("person_home_ownership_index")
      .setOutputCol("person_home_ownership_onehot")

    val pipeline1 = new Pipeline().setStages(Array(indexer1, encoder1))
    val model1 = pipeline1.fit(dfCreditRisk)
    val encodedDF1 = model1.transform(dfCreditRisk)

    val extractOneHot = udf((vec: SparseVector) => {
      val numCategories = vec.size
      (0 until numCategories).map(i => if (vec(i) == 1.0) 1 else 0).toArray
    })

    val homefinalDF = encodedDF1.withColumn("onehot_split", extractOneHot(col("person_home_ownership_onehot")))

    val oneHotColumns1 = (0 until encodedDF1.select("person_home_ownership_onehot").first().getAs[SparseVector]("person_home_ownership_onehot").size)
      .map(i => s"person_home_ownership_onehot_$i")

    val dfHome = oneHotColumns1.zipWithIndex.foldLeft(homefinalDF) { case (df, (colName, index)) =>
      df.withColumn(colName, col("onehot_split").getItem(index))
        .drop("person_home_ownership")
        .drop("person_home_ownership_index")
    }

    // One-hot encode 'loan_intent'
    val indexer2 = new StringIndexer()
      .setInputCol("loan_intent")
      .setOutputCol("loan_intent_index")

    val encoder2 = new OneHotEncoder()
      .setInputCol("loan_intent_index")
      .setOutputCol("loan_intent_onehot")

    val pipeline2 = new Pipeline().setStages(Array(indexer2, encoder2))
    val model2 = pipeline2.fit(dfHome)
    val encodedDF2 = model2.transform(dfHome)

    val intentfinalDF = encodedDF2.withColumn("onehot_split", extractOneHot(col("loan_intent_onehot")))

    val oneHotColumns2 = (0 until encodedDF2.select("loan_intent_onehot").first().getAs[SparseVector]("loan_intent_onehot").size)
      .map(i => s"loan_intent_onehot_$i")

    val finalDF = oneHotColumns2.zipWithIndex.foldLeft(intentfinalDF) { case (df, (colName, index)) =>
      df.withColumn(colName, col("onehot_split").getItem(index))
        .drop("loan_intent")
        .drop("loan_intent_index")
        .drop("person_home_ownership_index")
    }
    finalDF.printSchema()
    finalDF
  }

  def trainRandomForestModel(train: org.apache.spark.sql.DataFrame): PipelineModel = {
    // Specify feature columns
    val featureCols = Array(
      "person_age", "person_emp_length", "loan_amnt", "loan_int_rate",
      "loan_percent_income", "cb_person_cred_hist_length", "cb_person_default_on_file_binary",
      "remaining_income_to_receive", "remaining_employment_length", "age_to_history_length_ratio",
      "income_bracket", "credit_risk", "person_home_ownership_onehot_0",
      "person_home_ownership_onehot_1", "person_home_ownership_onehot_2",
      "loan_intent_onehot_0", "loan_intent_onehot_1", "loan_intent_onehot_2",
      "loan_intent_onehot_3", "loan_intent_onehot_4"
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

    // Train the model
    pipeline.fit(train)
  }
}
package services

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler

object LoanGradeTransformationPipeline {

  def transformData(df: Dataset[_], isTraining: Boolean = false)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    println("TransformData starting")
    df.printSchema()

    // Initial transformation with numeric mappings
    val initialDF = df
      // Convert basic numeric columns
      .withColumn("person_age", col("person_age").cast("integer"))
      .withColumn("person_income", col("person_income").cast("integer"))
      .withColumn("person_emp_length", col("person_emp_length").cast("double"))
      .withColumn("loan_amnt", col("loan_amnt").cast("integer"))
      .withColumn("loan_int_rate", col("loan_int_rate").cast("double"))
      .withColumn("loan_percent_income", col("loan_percent_income").cast("double"))
      .withColumn("cb_person_cred_hist_length", col("cb_person_cred_hist_length").cast("integer"))
      // Convert categorical to numeric
      .withColumn("home_ownership_num",
        when(col("person_home_ownership") === "RENT", 0)
          .when(col("person_home_ownership") === "MORTGAGE", 1)
          .when(col("person_home_ownership") === "OWN", 2)
          .otherwise(3))
      .withColumn("loan_intent_num",
        when(col("loan_intent") === "PERSONAL", 0)
          .when(col("loan_intent") === "EDUCATION", 1)
          .when(col("loan_intent") === "MEDICAL", 2)
          .when(col("loan_intent") === "VENTURE", 3)
          .when(col("loan_intent") === "HOMEIMPROVEMENT", 4)
          .otherwise(5))
      .withColumn("default_on_file_num",
        when(col("cb_person_default_on_file_binary") === "N", 0)
          .otherwise(1))

    // Add engineered features
    val enhancedDF = initialDF
      .withColumn("age_minus_history",
        log1p(col("person_age") - col("cb_person_cred_hist_length")).cast("double"))
      .withColumn("actual_loan_interest",
        (col("loan_amnt") * col("loan_int_rate")).cast("double"))
      .withColumn("employment_stability",
        when(col("person_emp_length") < 2, 1).otherwise(0))

    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length", "home_ownership_num",
        "loan_intent_num", "default_on_file_num",
        "age_minus_history", "actual_loan_interest", "employment_stability"
      ))
      .setOutputCol("features")
      .setHandleInvalid("keep")

    // Select columns based on whether it's training or prediction
    val finalDF = if (isTraining) {
      assembler.transform(enhancedDF)
        .select(
          col("person_age"),
          col("person_income"),
          col("person_emp_length"),
          col("loan_amnt"),
          col("loan_int_rate"),
          col("loan_percent_income"),
          col("cb_person_cred_hist_length"),
          col("loan_grade_num").cast("integer"),
          col("home_ownership_num").cast("integer"),
          col("loan_intent_num").cast("integer"),
          col("default_on_file_num").cast("integer"),
          col("age_minus_history"),
          col("actual_loan_interest"),
          col("employment_stability").cast("integer"),
          col("features")
        )
    } else {
      assembler.transform(enhancedDF)
        .select(
          col("person_age"),
          col("person_income"),
          col("person_emp_length"),
          col("loan_amnt"),
          col("loan_int_rate"),
          col("loan_percent_income"),
          col("cb_person_cred_hist_length"),
          col("home_ownership_num").cast("integer"),
          col("loan_intent_num").cast("integer"),
          col("default_on_file_num").cast("integer"),
          col("age_minus_history"),
          col("actual_loan_interest"),
          col("employment_stability").cast("integer"),
          col("features")
        )
    }

    println("Final schema:")
    finalDF.printSchema()
    println("done")
    finalDF
  }

  def transformTestData(df: Dataset[_])(implicit spark: SparkSession): DataFrame = {
    transformData(df, isTraining = false)
  }
}
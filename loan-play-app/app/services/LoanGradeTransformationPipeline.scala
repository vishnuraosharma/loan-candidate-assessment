package services

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction

object LoanGradeTransformationPipeline {

  def transformData(df: Dataset[_])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    // Drop unnecessary columns - modify this based on your input DataFrame
    val rawData = df.drop("id")

    println("TransformData starting")
    rawData.printSchema()

    // Define home ownership one-hot encoding
    val homeOwnershipDF = rawData
      .withColumn("homeOwnership_onehot_0",
        when(col("person_home_ownership") === "RENT", 1).otherwise(0))
      .withColumn("homeOwnership_onehot_1",
        when(col("person_home_ownership") === "MORTGAGE", 1).otherwise(0))
      .withColumn("homeOwnership_onehot_2",
        when(col("person_home_ownership") === "OWN", 1).otherwise(0))
      .drop("person_home_ownership")

    // Similarly for loan_intent
    val loanIntentDF = homeOwnershipDF
      .withColumn("loan_intent_onehot_0",
        when(col("loan_intent") === "EDUCATION", 1).otherwise(0))
      .withColumn("loan_intent_onehot_1",
        when(col("loan_intent") === "MEDICAL", 1).otherwise(0))
      .withColumn("loan_intent_onehot_2",
        when(col("loan_intent") === "PERSONAL", 1).otherwise(0))
      .withColumn("loan_intent_onehot_3",
        when(col("loan_intent") === "VENTURE", 1).otherwise(0))
      .withColumn("loan_intent_onehot_4",
        when(col("loan_intent") === "DEBTCONSOLIDATION", 1).otherwise(0))
      .drop("loan_intent")

    // Define cb_person_default_on_file one-hot encoding
    val finalDF = loanIntentDF
      .withColumn("defaultOnFile_onehot_0",
        when(col("cb_person_default_on_file_binary") === "Y", 1).otherwise(0))
      .withColumn("defaultOnFile_onehot_1",
        when(col("cb_person_default_on_file_binary") === "N", 1).otherwise(0))
      .drop("cb_person_default_on_file_binary")
      .drop("loan_grade")

    println("Final schema:")
    finalDF.printSchema()
    println("done")
    finalDF
  }
}
package services

import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction

object LoanStatusTransformationPipeline {

  def transformData(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits._

    println("TransformData starting")
    df.printSchema()

    // Add calculated columns
    val updatedDF = df
      .withColumn("remaining_income_to_receive", col("person_income") * (lit(80) - col("person_age")))
      .withColumn("remaining_employment_length", lit(65) - col("person_age"))
      .withColumn("age_to_history_length_ratio", col("person_age") / col("cb_person_cred_hist_length"))
      .withColumn("credit_risk", (col("loan_amnt") * col("loan_int_rate")) / col("cb_person_cred_hist_length"))

    // Define income bucket UDF
    val incomeBuckets: UserDefinedFunction = udf((income: Int) => income match {
      case i if i < 30000 => 1
      case i if i >= 30000 && i < 50000 => 2
      case i if i >= 50000 && i < 70000 => 3
      case i if i >= 70000 && i < 100000 => 4
      case i if i >= 100000 && i < 150000 => 5
      case _ => 6
    })

    val incomeBracketedDF = updatedDF.withColumn("income_bracket", incomeBuckets(col("person_income")))

    // One-hot encode person_home_ownership
    val oneHotDF = incomeBracketedDF
      .withColumn("person_home_ownership_onehot_0",
        when(col("person_home_ownership") === "RENT", 1).otherwise(0))
      .withColumn("person_home_ownership_onehot_1",
        when(col("person_home_ownership") === "MORTGAGE", 1).otherwise(0))
      .withColumn("person_home_ownership_onehot_2",
        when(col("person_home_ownership") === "OWN", 1).otherwise(0))
      .drop("person_home_ownership")

    // Similarly for loan_intent
    val onehotterDF = oneHotDF
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

    // add a col called loan_intent_onehot with all 0s as placeholder
    val finalDF = onehotterDF.withColumn("loan_intent_onehot", lit(0))

    println("Final schema:")
    finalDF.printSchema()
    println("ashjkjfds")
    finalDF.select(
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
  }
}

// Databricks notebook source
// MAGIC %md
// MAGIC # Modeling Loan Status
// MAGIC Author: Vishnu Rao-Sharma
// MAGIC
// MAGIC In this notebook we will use Scala Spark to explore Kaggle's Loan Approval dataset. 
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

var train = spark.read.format(fileType)
  .option("inferSchema", inferSchema)
  .option("header", firstRowIsHeader)
  .option("sep", delimiter)
  .load(fileLocation)

train.show()  

// COMMAND ----------

// MAGIC %md
// MAGIC Let's drop `loan_grade` and the `id`

// COMMAND ----------

train = train.drop("loan_grade")
train = train.drop("id")


// COMMAND ----------

{
  // Number of rows and columns
  val numRows = train.count()
  val numCols = train.columns.length
  println(s"Number of rows: $numRows \nNumber of cols: $numCols\n")

  // Count of negative and positive loan statuses
  val negative = train.filter($"loan_status" === 0).count()
  val positive = train.filter($"loan_status" === 1).count()
  println(s"Rejected: $negative\nApproved: $positive \n%Approved: ${positive.toDouble / (positive + negative)}")
}


// COMMAND ----------

train.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC Here we can see the shape of our dataset, with ~59K rows and 11 columns. 
// MAGIC
// MAGIC Since we are prediction Loan Status with this notebook, we show the % of approved applications. At 14%, it's pretty clear that it's not easy to be approved. Let's look at how profile markers correlate with `loan_status` to understand the most important aspects of getting approved.
// MAGIC

// COMMAND ----------

train.columns

// COMMAND ----------

// MAGIC %md
// MAGIC ## Exploratory Data Analysis

// COMMAND ----------

val missingValues = train.columns.map(c => (c, train.filter(train(c).isNull).count()))
missingValues.foreach{ case (col, count) => println(s"$col has $count missing values") }

// COMMAND ----------

// MAGIC %md
// MAGIC No missing values in our dataset. Let's go ahead and create a correlation matrix for our Numeric cols.

// COMMAND ----------


import org.apache.spark.sql.functions._
import spark.implicits._ // Required for toDF()

// Select numeric columns from the DataFrame
val numericCols = train.schema.fields.collect {
  case field if field.dataType.typeName == "double" || field.dataType.typeName == "integer" => field.name
}

// Compute correlation values
val correlationValues = for {
  col1 <- numericCols
  col2 <- numericCols if col1 < col2 // This condition ensures col1 is always less than col2 lexicographically
} yield (col1, col2, train.stat.corr(col1, col2))

// Convert correlation values to a DataFrame
val correlationDF = correlationValues.toSeq.toDF("Column1", "Column2", "Correlation")

// Filter the DataFrame to exclude correlations equal to 1
val filteredDF = correlationDF.filter(abs(col("Correlation")) < 1)

// Sort the DataFrame by the absolute correlation values in descending order
val dfSorted = filteredDF.orderBy(abs(col("Correlation")).desc)

// Show the top correlations
dfSorted.show(40)


// COMMAND ----------

// MAGIC %md
// MAGIC **Postive Correlations**
// MAGIC - `cb_person_cred_hist_length` and `person_age` (0.87): This indicates a strong positive relationship between these two variables. Older individuals seem to have a longer credit history which makes sense.
// MAGIC - `loan_amnt` and `loan_percent_income` (0.65): A moderately strong positive correlation, suggesting that higher loan amounts are associated with a higher percentage of income being allocated to the loan.
// MAGIC - `loan_int_rate` and `loan_status` (0.34): A moderate positive correlation, meaning that higher loan interest rates are somewhat associated with a higher probability of default or a worse loan status.
// MAGIC
// MAGIC **Negative Correlations**
// MAGIC - `loan_percent_income` and `person_income` (-0.28): A moderate negative relationship, which may indicate that individuals with higher incomes are taking smaller loans relative to their income.
// MAGIC - `loan_status` and `person_income` (-0.17): A slight negative correlation, suggesting that people with higher incomes might be less likely to have loan defaults or poor loan statuses.
// MAGIC - `loan_int_rate` and `person_emp_length` (-0.10): A slight negative correlation, which may imply that individuals with longer employment histories tend to get lower interest rates.

// COMMAND ----------

// MAGIC %md
// MAGIC Let's drill down into age.

// COMMAND ----------

import org.apache.spark.sql.functions._

// Define age buckets
val ageBuckets = udf((age: Int) => age match {
  case a if a < 25 => "Under 25"
  case a if a >= 25 && a < 35 => "25-34"
  case a if a >= 35 && a < 45 => "35-44"
  case a if a >= 45 && a < 55 => "45-54"
  case a if a >= 55 && a < 65 => "55-64"
  case _ => "65+"
})

// Add a new column for age buckets
val trainWithAgeBuckets = train.withColumn("age_bucket", ageBuckets(col("person_age")))

// Group by age bucket and calculate the average loan status rate
val result = trainWithAgeBuckets
  .groupBy("age_bucket")
  .agg(avg("loan_status").alias("average_loan_status_rate"))
  .orderBy("age_bucket")

// Show the result
result.show()


// COMMAND ----------

// MAGIC %md
// MAGIC This drilldown shows us that bucketing age is proabably not necessary because loan_status approval moves in step with age and there really isn't a steep climb between buckets.

// COMMAND ----------

// MAGIC %md
// MAGIC Let's drill down into income.

// COMMAND ----------

import org.apache.spark.sql.functions._

// Define income buckets
val incomeBuckets = udf((income: Int) => income match {
  case i if i < 30000 => "Under 30k"
  case i if i >= 30000 && i < 50000 => "30k-49k"
  case i if i >= 50000 && i < 70000 => "50k-69k"
  case i if i >= 70000 && i < 100000 => "70k-99k"
  case i if i >= 100000 && i < 150000 => "100k-149k"
  case _ => "150k+"
})

// Add a new column for income buckets
val trainWithIncomeBuckets = train.withColumn("income_bucket", incomeBuckets(col("person_income")))

// Group by income bucket and calculate the average loan status rate
val result = trainWithIncomeBuckets
  .groupBy("income_bucket")
  .agg(avg("loan_status").alias("average_loan_status_rate"))
  .orderBy("income_bucket")

// Show the result
result.orderBy((col("average_loan_status_rate")).desc).show()


// COMMAND ----------

// MAGIC %md
// MAGIC Although one might assume Loans would be granted to those with higher incomes, the opposite seems to be true. Perhaps Loan grantors are employing more "predatory" tactics and count on low income applicants to default and stay in debt. This is a good column to turn into a categorical feature because of the steep drop off between each tier. 

// COMMAND ----------

// MAGIC %md
// MAGIC Let's look at outliers for a couple numeric columns "person_income", "loan_amnt". We'll focus on these two because these are financial numbers that, unlike Age, may be extremely high and skew the prediction. Let's find outliers using the IQR. 

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

// Select numeric columns
val numericCols = Seq( "person_income",  
                      "loan_amnt")

// Function to calculate IQR bounds
def getIQRBounds(df: DataFrame, column: String) = {
  val quantiles = df.stat.approxQuantile(column, Array(0.25, 0.75), 0.0)
  val q1 = quantiles(0)
  val q3 = quantiles(1)
  val iqr = q3 - q1
  val lowerBound = q1 - 1.5 * iqr
  val upperBound = q3 + 1.5 * iqr
  (lowerBound, upperBound)
}

// Create a DataFrame with outlier flags
val outliersDF = numericCols.foldLeft(train) { (df, colName) =>
  val (lowerBound, upperBound) = getIQRBounds(df, colName)

  // Add a column that flags outliers (True for outliers)
  df.withColumn(s"${colName}_outlier", 
    when(col(colName) < lowerBound || col(colName) > upperBound, true).otherwise(false))
}

// Count the number of outliers for each column
val outlierCounts = numericCols.map { colName =>
  val outlierCol = s"${colName}_outlier"
  
  // Count the number of true values (outliers) in the outlier column
  val count = outliersDF.filter(col(outlierCol) === true).count()
  (colName, count)
}

// Display the counts
outlierCounts.foreach { case (colName, count) =>
  println(s"Column: $colName, Number of outliers: $count")
}


// COMMAND ----------

// MAGIC %md
// MAGIC For these columns, we should handle outliers by imputing values with the Median for each age bracket.

// COMMAND ----------

// MAGIC %md
// MAGIC For the next analysis, let's explore the relationship between person_home_ownership (e.g., whether the person owns, rents, or has a different home ownership status) and loan_status. This can give insights into whether homeownership status has an impact on loan approval.

// COMMAND ----------

// Import necessary functions
import org.apache.spark.sql.functions._

// Calculate the total count for each person_home_ownership group
val totalByHomeOwnership = train
  .groupBy("person_home_ownership")
  .agg(
    count("loan_status").alias("total_count")
  )

// Group by person_home_ownership and loan_status, count the occurrences
val homeOwnershipByLoanStatus = train
  .groupBy("person_home_ownership", "loan_status")
  .agg(
    count("loan_status").alias("count")
  )

// Join the counts with the total counts to calculate percentages
val homeOwnershipWithPercentages = homeOwnershipByLoanStatus
  .join(totalByHomeOwnership, "person_home_ownership")
  .withColumn("percentage", col("count") / col("total_count") * 100)
  .orderBy("person_home_ownership", "loan_status")

// Show the result with counts and percentages
homeOwnershipWithPercentages.show()


// COMMAND ----------

// MAGIC %md
// MAGIC The majority of all groups are denied loans with MORTGAGE holders having the highest rate of loan denial, followed by OWN (homeowners).
// MAGIC Renters have the highest percentage of approved loans (22.26%), while MORTGAGE holders have the lowest approval rate.
// MAGIC The OTHER category has a very small sample size, so conclusions from this group may not be as reliable.
// MAGIC
// MAGIC - Homeownership status (`person_home_ownership`) appears to be an important feature that impacts loan approval, especially with mortgage holders and homeowners having very low approval rates.
// MAGIC - Renters seem to have a higher likelihood of loan approval, which might be a useful factor in a predictive model.

// COMMAND ----------

// Group by loan_intent and loan_status to count the occurrences
val loanIntentByLoanStatus = train
  .groupBy("loan_intent", "loan_status")
  .agg(
    count("loan_status").alias("count")
  )
  .orderBy("loan_intent", "loan_status")

// Calculate total counts for each loan_intent category
val totalByLoanIntent = train
  .groupBy("loan_intent")
  .agg(
    count("loan_status").alias("total_count")
  )

// Join the counts with total counts to calculate percentages
val loanIntentWithPercentages = loanIntentByLoanStatus
  .join(totalByLoanIntent, "loan_intent")
  .withColumn("percentage", col("count") / col("total_count") * 100)
  .orderBy("loan_intent", "loan_status")

// Show the result with counts and percentages
loanIntentWithPercentages.show()


// COMMAND ----------

// MAGIC %md
// MAGIC Venture loans and education loans show the highest denial rates, while medical loans have the lowest denial rate, but still face a high percentage of denials (82%).
// MAGIC Debt consolidation and home improvement loans also face relatively high denial rates, around 82-83%.
// MAGIC - `loan_intent` is an important feature for understanding loan approval. Certain loan intents, such as venture and education, have higher denial rates, while others like medical or personal might have slightly better approval chances.
// MAGIC - Venture loans have the highest denial rate, suggesting that people applying for business-related loans may have a harder time getting approval compared to other types of loans. This makes sense as Ventures are a bit riskier and are not as essential as Medical or Education Loans.

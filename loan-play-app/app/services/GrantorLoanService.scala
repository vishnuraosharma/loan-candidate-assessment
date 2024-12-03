package services

import org.apache.spark.sql.{SparkSession, DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import models.Loan
import org.apache.spark.serializer.KryoSerializer

class GrantorLoanService {
  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  def processLoan(loan: Loan): DataFrame = {
    // Convert single loan to DataFrame with required schema
    val loanDF = Seq((
      loan.id,
      loan.personAge,
      loan.personIncome,
      loan.ownershipType,
      loan.employmentLength,
      loan.intent,
      loan.amountRequested,
      loan.interestRate,
      loan.amountRequested / loan.personIncome.toDouble,
      if (loan.priorDefault) 1 else 0,
      loan.creditHistory
    )).toDF(
      "id", "person_age", "person_income", "person_home_ownership", "person_emp_length",
      "loan_intent", "loan_amnt",
      "loan_int_rate", "loan_percent_income",
      "cb_person_default_on_file_binary", "cb_person_cred_hist_length")

    // Transform the data using the pipeline and convert to DataFrame
    LoanStatusTransformationPipeline.transformData(loanDF).toDF()
  }
}
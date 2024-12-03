package services

import org.apache.spark.sql.{SparkSession, DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import models.Loan
import org.apache.spark.serializer.KryoSerializer

class LoanGradeService {
  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  def processLoan(loan: Loan): DataFrame = {
    // Convert single loan to DataFrame with required schema

  }
}
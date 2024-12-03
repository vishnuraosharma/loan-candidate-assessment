package services

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.sql.SparkSession

object LoanGradeTransformationPipeline {

  def transformData(df: Dataset[_])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    // Drop unnecessary columns - modify this based on your input DataFrame
    val rawData = df.drop("id")

    println("TransformData starting")
    rawData.printSchema()

    // Index and then OneHotEncode features

    // home_ownership
    val homeOwnershipIndexer = new StringIndexer()
      .setInputCol("person_home_ownership")
      .setOutputCol("homeOwnershipIndex")
      .fit(rawData)
    val homeOwnershipEncoder = new OneHotEncoder()
      .setInputCol("homeOwnershipIndex")
      .setOutputCol("homeOwnershipVec")

    // loan_intent
    val loanIntentIndexer = new StringIndexer()
      .setInputCol("loan_intent")
      .setOutputCol("loanIntentIndex")
      .fit(rawData)
    val loanIntentEncoder = new OneHotEncoder()
      .setInputCol("loanIntentIndex")
      .setOutputCol("loanIntentVec")

    // cb_person_default_on_file
    val defaultOnFileIndexer = new StringIndexer()
      .setInputCol("cb_person_default_on_file")
      .setOutputCol("defaultOnFileIndex")
      .fit(rawData)
    val defaultOnFileEncoder = new OneHotEncoder()
      .setInputCol("defaultOnFileIndex")
      .setOutputCol("defaultOnFileVec")

    val assembler = new VectorAssembler()
      .setInputCols(Array("homeOwnershipVec", "loanIntentVec", "defaultOnFileVec", "person_age",
        "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"))
      .setOutputCol("features")

    // Pipeline stages
    val pipeline = new Pipeline().setStages(Array(homeOwnershipIndexer, homeOwnershipEncoder,
      loanIntentIndexer, loanIntentEncoder, defaultOnFileIndexer, defaultOnFileEncoder,
      assembler))

    // Fit the pipeline
    val model = pipeline.fit(rawData)
    val transformedData = model.transform(rawData)

    val finalDF = transformedData.drop("person_home_ownership")
      .drop("loan_intent")
      .drop("cb_person_default_on_file")
      .drop("loan_grade")

    println("Final schema:")
    finalDF.printSchema()

    finalDF
  }
}
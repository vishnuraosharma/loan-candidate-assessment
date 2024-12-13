import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.CrossValidatorModel
import java.nio.file.{Files, Path}

class LoanGraderSpec extends AnyFunSpec with Matchers {

  describe("LoanGrader") {
    it("should train and evaluate model successfully") {
      // Arrange
      val spark = SparkSession.builder
        .appName("Loan Grader Test")
        .master("local[*]")
        .getOrCreate()

      import spark.implicits._

      // Create test training data
      val trainData = Seq(
        (1, 25, 50000, "RENT", 3.0, "PERSONAL", 10000, 0.1, "A", 0.2, "N", 5),
        (2, 35, 75000, "MORTGAGE", 5.0, "EDUCATION", 15000, 0.15, "B", 0.2, "Y", 8),
        (3, 45, 100000, "OWN", 10.0, "MEDICAL", 20000, 0.12, "C", 0.15, "N", 12)
      ).toDF(
        "id",
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_amnt",
        "loan_int_rate",
        "loan_grade",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length"
      )

      // Create test data
      val testData = Seq(
        (1, 30, 60000, "RENT", 4.0, "PERSONAL", 12000, 0.11, 0.2, "N", 6),
        (2, 40, 85000, "MORTGAGE", 6.0, "EDUCATION", 17000, 0.14, 0.2, "Y", 9)
      ).toDF(
        "id",
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length"
      )

      // Create temporary directories for data
      val tempDir = Files.createTempDirectory("loan_grader_test")
      val trainPath = tempDir.resolve("train.csv")
      val testPath = tempDir.resolve("test.csv")
      val modelPath = tempDir.resolve("model")

      try {
        // Save test data to temporary files
        trainData.write.option("header", true).csv(trainPath.toString)
        testData.write.option("header", true).csv(testPath.toString)

        // Act
        val args = Array(
          trainPath.toString,
          testPath.toString,
          modelPath.toString
        )

        noException should be thrownBy {
          LoanGrader.main(args)
        }

        // Assert
        Files.exists(modelPath) shouldBe true

        // Load and verify the saved model
        val loadedModel = CrossValidatorModel.load(modelPath.toString)
        loadedModel should not be null

        // Optional: Make predictions with loaded model
        val predictions = loadedModel.transform(testData)
        predictions.select("prediction").count() shouldBe testData.count()

      } finally {
        // Clean up
        deleteRecursively(tempDir)
        spark.stop()
      }
    }

    it("should handle invalid input gracefully") {
      val args = Array("nonexistent/path/train.csv", "nonexistent/path/test.csv", "nonexistent/path/model")

      val thrown = intercept[Exception] {
        LoanGrader.main(args)
      }

      thrown.getMessage should include("Error")
    }
  }

  // Helper method to recursively delete directory
  private def deleteRecursively(path: Path): Unit = {
    if (Files.exists(path)) {
      if (Files.isDirectory(path)) {
        Files.list(path).forEach(deleteRecursively)
      }
      Files.delete(path)
    }
  }
}
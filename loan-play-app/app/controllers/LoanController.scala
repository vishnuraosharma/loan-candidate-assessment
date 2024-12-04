package controllers

import javax.inject._
import play.api.mvc._
import play.api.data._
import play.api.data.Forms._
import play.api.data.format.Formats._
import play.api.i18n.I18nSupport
import java.util.UUID
import models.{Loan, LoanFormData}
import services.GrantorLoanService
import services.LoanGradeService
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

@Singleton
class LoanController @Inject()(
                                cc: MessagesControllerComponents,
                                grantorLoanService: GrantorLoanService,
                                gradeService:LoanGradeService,
                                env: play.api.Environment
                              ) extends MessagesAbstractController(cc) with I18nSupport {


  // Map of username -> their loans
  private val userLoans = scala.collection.mutable.Map[String, scala.collection.mutable.ArrayBuffer[Loan]]()
  
  // Store DataFrames for each grantor
  private val grantorDataFrames = scala.collection.mutable.Map[String, DataFrame]()

  val loanForm = Form(
    mapping(
      "personAge" -> number,
      "personIncome" -> number,
      "ownershipType" -> nonEmptyText,
      "employmentLength" -> number,
      "intent" -> nonEmptyText,
      "amountRequested" -> of[Double],
      "interestRate" -> of[Double],
      "priorDefault" -> boolean,
      "creditHistory" -> number
    )(LoanFormData.apply)(LoanFormData.unapply)
  )

  def manageLoanPage = Action { implicit request =>
    request.session.get("username") match {
      case Some(username) => 
        val loans = userLoans.getOrElseUpdate(username, scala.collection.mutable.ArrayBuffer[Loan]())
        Ok(views.html.manageLoans(loanForm, loans.toSeq))
      case None => Redirect(routes.AuthController.login)
    }
  }

  def createLoan = Action { implicit request =>
    request.session.get("username") match {
      case Some(username) =>
        loanForm.bindFromRequest.fold(
          formWithErrors => {
            val loans = userLoans.getOrElseUpdate(username, scala.collection.mutable.ArrayBuffer[Loan]())
            BadRequest(views.html.manageLoans(formWithErrors, loans.toSeq))
          },
          loanData => {
            val loan = Loan(
              id = UUID.randomUUID().toString,
              personAge = loanData.personAge,
              personIncome = loanData.personIncome,
              ownershipType = loanData.ownershipType,
              employmentLength = loanData.employmentLength,
              intent = loanData.intent,
              amountRequested = loanData.amountRequested,
              interestRate = loanData.interestRate,
              priorDefault = loanData.priorDefault,
              creditHistory = loanData.creditHistory,
              grantorUsername = username
            )
            
            // Process loan through Spark pipeline
            val transformedDF = grantorLoanService.processLoan(loan)
            val gradeDF = gradeService.processLoan(loan)
            //val gradeDF = loanGradeService.processLoan(loan)
            // Print the id of the loan
            println(s"Loan ID: ${loan.id}")
            //print the id column of the transformedDF
            transformedDF.show()

            // Update both maps
            userLoans.getOrElseUpdate(username, scala.collection.mutable.ArrayBuffer[Loan]()) += loan
            
            grantorDataFrames.get(username) match {
              case Some(existingDF) =>
                grantorDataFrames(username) = existingDF.union(transformedDF)
              case None =>
                grantorDataFrames(username) = transformedDF
            }
            
            Redirect(routes.DashboardController.index)
              .flashing("success" -> "Loan application submitted successfully")
          }
        )
      case None => Redirect(routes.AuthController.login)
    }
  }

  def getUserLoans(username: String): Seq[Loan] = {
    userLoans.getOrElseUpdate(username, scala.collection.mutable.ArrayBuffer[Loan]()).toSeq
  }

  def calculateLoanGrade(id: String) = Action { implicit request =>
    request.session.get("username") match {
      case Some(username) =>
        val userLoansList = userLoans.getOrElseUpdate(username, scala.collection.mutable.ArrayBuffer[Loan]())
        userLoansList.find(_.id == id) match {
          case Some(loan) =>
            try {
              // Process the loan through the grade service to get the feature vector
              val gradeDF = gradeService.processLoan(loan)
              //val statusDF = grantorLoanService.processLoan(loan)
              
              // Load the saved model
              //C:\Users\momog\Desktop\loan-candidate-assessment\loan-play-app\model\loan_grader_model\metadata
              val model = CrossValidatorModel.load("C:\\Users\\momog\\Desktop\\loan-candidate-assessment\\loan-play-app\\model\\loan_grader_model")
              // Load the status model
              //val statusmodel = PipelineModel.load("C:\\Users\\momog\\Desktop\\loan-candidate-assessment\\loan-play-app\\model\\loan_status_model")

              // Make prediction
              val prediction = model.transform(gradeDF)
              //val statusprediction = statusmodel.transform(statusDF)

              // Extract the predicted grade (assuming it's in the 'prediction' column)
              val predictedGrade = prediction.select("prediction").first().getDouble(0)
              //val predictedStatus = statusprediction.select("prediction").first().getDouble(0)
              
              // Convert numeric prediction to letter grade if needed
              val letterGrade = predictedGrade match {
                case 0 => "A"
                case 1 => "B"
                case 2 => "C"
                case 3 => "D"
                case 4 => "E"
                case 5 => "F"
                case _ => "G"
              }
              println(letterGrade)
              //println(predictedStatus)
              Redirect(routes.DashboardController.index())
                .flashing("success" -> s"Loan grade calculated: $letterGrade")
                
            } catch {
              case e: Exception =>
                e.printStackTrace()
                Redirect(routes.DashboardController.index())
                  .flashing("error" -> "Error calculating loan grade")
            }
              
          case None =>
            Redirect(routes.DashboardController.index())
              .flashing("error" -> "Loan not found")
        }
      case None => 
        Redirect(routes.AuthController.login)
    }
  }

  def getGrantorDataFrame(grantorUsername: String): Option[DataFrame] = {
    grantorDataFrames.get(grantorUsername)
  }
}
package controllers

import javax.inject._
import play.api.mvc._
import play.api.data._
import play.api.data.Forms._
import play.api.data.format.Formats._
import play.api.i18n.I18nSupport
import java.util.UUID
import models.{Loan, LoanFormData}
import services.LoanStatusService
import services.LoanGradeService
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

@Singleton
class LoanController @Inject()(
                                cc: MessagesControllerComponents,
                                grantorLoanService: LoanStatusService,
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
            
            // Update user loans map
            userLoans.getOrElseUpdate(username, scala.collection.mutable.ArrayBuffer[Loan]()) += loan
            
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
              //val gradeDF = gradeService.processLoan(loan)
              val statusDF = grantorLoanService.processLoan(loan)

              
              // Load the grade model
              //val gradermodelPath = env.getFile("model/loan_grader_model").getAbsolutePath
              //val grademodel = CrossValidatorModel.load(gradermodelPath)
             //println("Loan Grade Model Loaded")

              // Load the status model
              val statusmodelPath = env.getFile("model/loan_status_model").getAbsolutePath
              val statusmodel = PipelineModel.load(statusmodelPath)
              println("Loan Status Model Loaded")

              // Make prediction
              val gradeprediction = grademodel.transform(gradeDF)
              val statusprediction = statusmodel.transform(statusDF)
              println("Loan Status Prediction Made")

              // Extract the predicted grade (assuming it's in the 'prediction' column)
              val predictedGrade = gradeprediction.select("prediction").first().getDouble(0)
              val predictedStatus = statusprediction.select("prediction").first().getDouble(0)

              // Convert numeric prediction to readable output
              val finalStatus = predictedStatus match {
                case 0 => "Rejected"
                case _ => "Approved"
              }

//              val letterGrade = predictedGrade match {
//                case 0 => "A"
//                case 1 => "B"
//                case 2 => "C"
//                case 3 => "D"
//                case 4 => "E"
//                case 5 => "F"
//                case _ => "G"
//              }
              //println(letterGrade)
              println(finalStatus)
              Redirect(routes.DashboardController.index())
                .flashing("success" -> s"Loan status calculated: $finalStatus")
                
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
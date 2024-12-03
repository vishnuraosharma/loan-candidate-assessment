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
import org.apache.spark.sql.DataFrame

@Singleton
class LoanController @Inject()(
                                cc: MessagesControllerComponents,
                                grantorLoanService: GrantorLoanService
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
            // TODO: Implement actual loan grade calculation logic
            // For now, just redirect back with a flash message
            Redirect(routes.DashboardController.index())
              .flashing("info" -> "Loan grade calculation will be implemented soon")
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
package controllers

import javax.inject._
import play.api.mvc._
import play.api.data._
import play.api.data.Forms._
import play.api.data.format.Formats._
import play.api.i18n.I18nSupport
import java.util.UUID
import models.{Loan, LoanFormData}

@Singleton
class LoanController @Inject()(
                                cc: MessagesControllerComponents
                              ) extends MessagesAbstractController(cc) with I18nSupport {

  // Map of username -> their loans
  private val userLoans = scala.collection.mutable.Map[String, scala.collection.mutable.ArrayBuffer[Loan]]()

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
}
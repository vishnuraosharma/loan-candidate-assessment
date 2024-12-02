package controllers

import javax.inject._
import play.api.mvc._
import play.api.i18n.I18nSupport

@Singleton
class DashboardController @Inject()(
                                     cc: MessagesControllerComponents,
                                     loanController: LoanController
                                   ) extends MessagesAbstractController(cc) with I18nSupport {

  def index = Action { implicit request =>
    request.session.get("role") match {
      case Some("Loan Grantor") => 
        val username = request.session.get("username").get
        val loans = loanController.getUserLoans(username)
        Ok(views.html.grantorDashboard(loans))
      case Some("Loan Requester") =>
        val username = request.session.get("username").get
        val loans = loanController.getUserLoans(username)
        Ok(views.html.requesterDashboard(loanController.loanForm, loans))
      case _ => Redirect(routes.AuthController.login)
    }
  }
}
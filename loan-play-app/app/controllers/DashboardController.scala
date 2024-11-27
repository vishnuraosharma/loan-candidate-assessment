package controllers

import javax.inject._
import play.api.mvc._
import play.api.i18n.I18nSupport

@Singleton
class DashboardController @Inject()(
                                     cc: MessagesControllerComponents
                                   ) extends MessagesAbstractController(cc) with I18nSupport {

  def index = Action { implicit request =>
    request.session.get("role") match {
      case Some("Loan Grantor") => Ok(views.html.grantorDashboard())
      case Some("Loan Requester") => Ok(views.html.requesterDashboard())
      case _ => Redirect(routes.AuthController.login)
    }
  }
}
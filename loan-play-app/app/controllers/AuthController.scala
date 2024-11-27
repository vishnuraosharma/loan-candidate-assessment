package controllers

import javax.inject._
import play.api.mvc._
import play.api.data._
import play.api.data.Forms._
import play.api.i18n.I18nSupport

@Singleton
class AuthController @Inject()(
                                cc: MessagesControllerComponents
                              ) extends MessagesAbstractController(cc) with I18nSupport {

  // In-memory storage (replace with database in production)
  private val users = scala.collection.mutable.Map[String, User]()

  // Form definitions
  val loginForm = Form(
    mapping(
      "username" -> nonEmptyText,
      "password" -> nonEmptyText
    )(LoginData.apply)(LoginData.unapply)
  )

  val signupForm = Form(
    mapping(
      "username" -> nonEmptyText,
      "password" -> nonEmptyText,
      "role" -> nonEmptyText
    )(SignupData.apply)(SignupData.unapply)
  )

  def login = Action { implicit request =>
    Ok(views.html.login(loginForm))
  }

  def doLogin = Action { implicit request =>
    loginForm.bindFromRequest.fold(
      formWithErrors => BadRequest(views.html.login(formWithErrors)),
      loginData => {
        users.get(loginData.username) match {
          case Some(user) if user.password == loginData.password =>
            Redirect(routes.DashboardController.index)
              .withSession("username" -> user.username, "role" -> user.role)
          case _ =>
            Redirect(routes.AuthController.login)
              .flashing("error" -> "Invalid username or password")
        }
      }
    )
  }

  def signup = Action { implicit request =>
    Ok(views.html.signup(signupForm))
  }

  def doSignup = Action { implicit request =>
    signupForm.bindFromRequest.fold(
      formWithErrors => BadRequest(views.html.signup(formWithErrors)),
      signupData => {
        if (users.contains(signupData.username)) {
          Redirect(routes.AuthController.signup)
            .flashing("error" -> "Username already exists")
        } else {
          users(signupData.username) = User(
            signupData.username,
            signupData.password,
            signupData.role
          )
          Redirect(routes.AuthController.login)
            .flashing("success" -> "Sign up successful!")
        }
      }
    )
  }

  def logout = Action {
    Redirect(routes.AuthController.login).withNewSession
  }
}

case class LoginData(username: String, password: String)
case class SignupData(username: String, password: String, role: String)
case class User(username: String, password: String, role: String)
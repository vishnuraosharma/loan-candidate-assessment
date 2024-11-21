import scala.swing._
import scala.swing.event._

import scala.collection.mutable

object LoanApp extends SimpleSwingApplication {

  // In-memory database for users
  val users: mutable.Map[String, (String, String)] = mutable.Map() // (username -> (password, role))

  // Entry point of the application
  def top: Frame = new MainFrame {
    title = "Loan Management System"
    preferredSize = new Dimension(400, 300)

    val loginButton = new Button("Login")
    val signUpButton = new Button("Sign Up")

    contents = new BoxPanel(Orientation.Vertical) {
      contents += new Label("Welcome to Loan Management System")
      contents += Swing.VStrut(20)
      contents += loginButton
      contents += Swing.VStrut(10)
      contents += signUpButton
      border = Swing.EmptyBorder(30, 30, 30, 30)
    }

    listenTo(loginButton, signUpButton)

    reactions += {
      case ButtonClicked(`loginButton`) =>
        openLoginScreen()
      case ButtonClicked(`signUpButton`) =>
        openSignUpScreen()
    }
  }

  def openLoginScreen(): Unit = {
    val usernameField = new TextField(20)
    val passwordField = new PasswordField(20)
    val loginButton = new Button("Login")

    val loginFrame = new Frame {
      title = "Login"
      contents = new BoxPanel(Orientation.Vertical) {
        contents += new Label("Username:")
        contents += usernameField
        contents += new Label("Password:")
        contents += passwordField
        contents += Swing.VStrut(10)
        contents += loginButton
        border = Swing.EmptyBorder(20, 20, 20, 20)
      }

      listenTo(loginButton)

      reactions += {
        case ButtonClicked(`loginButton`) =>
          val username = usernameField.text
          val password = passwordField.password.mkString

          users.get(username) match {
            case Some((storedPassword, role)) if storedPassword == password =>
              role match {
                case "Loan Grantor"   => openGrantorScreen()
                case "Loan Requester" => openRequesterScreen()
              }
              close()
            case _ =>
              Dialog.showMessage(this, "Invalid username or password", "Error", Dialog.Message.Error)
          }
      }

      centerOnScreen()
      open()
    }
  }

  def openSignUpScreen(): Unit = {
    val usernameField = new TextField(20)
    val passwordField = new PasswordField(20)
    val roleSelection = new ComboBox(Seq("Loan Grantor", "Loan Requester"))
    val signUpButton = new Button("Sign Up")

    val signUpFrame = new Frame {
      title = "Sign Up"
      contents = new BoxPanel(Orientation.Vertical) {
        contents += new Label("Username:")
        contents += usernameField
        contents += new Label("Password:")
        contents += passwordField
        contents += new Label("Role:")
        contents += roleSelection
        contents += Swing.VStrut(10)
        contents += signUpButton
        border = Swing.EmptyBorder(20, 20, 20, 20)
      }

      listenTo(signUpButton)

      reactions += {
        case ButtonClicked(`signUpButton`) =>
          val username = usernameField.text
          val password = passwordField.password.mkString
          val role = roleSelection.selection.item

          if (users.contains(username)) {
            Dialog.showMessage(this, "Username already exists", "Error", Dialog.Message.Error)
          } else if (username.isEmpty || password.isEmpty) {
            Dialog.showMessage(this, "Fields cannot be empty", "Error", Dialog.Message.Error)
          } else {
            users(username) = (password, role)
            Dialog.showMessage(this, "Sign up successful!", "Success", Dialog.Message.Info)
            close()
          }
      }

      centerOnScreen()
      open()
    }
  }

  def openGrantorScreen(): Unit = {
    new Frame {
      title = "Loan Grantor Dashboard"
      contents = new BoxPanel(Orientation.Vertical) {
        contents += new Label("Welcome, Loan Grantor!")
        contents += Swing.VStrut(20)
        contents += new Button("Manage Loans")
        border = Swing.EmptyBorder(20, 20, 20, 20)
      }
      centerOnScreen()
      open()
    }
  }

  def openRequesterScreen(): Unit = {
    new Frame {
      title = "Loan Requester Dashboard"
      contents = new BoxPanel(Orientation.Vertical) {
        contents += new Label("Welcome, Loan Requester!")
        contents += Swing.VStrut(20)
        contents += new Button("Request Loans")
        border = Swing.EmptyBorder(20, 20, 20, 20)
      }
      centerOnScreen()
      open()
    }
  }
}

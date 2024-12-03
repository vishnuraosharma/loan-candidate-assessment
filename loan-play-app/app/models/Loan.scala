package models

case class Loan(
                 id: String,
                 personAge: Int,
                 personIncome: Int,
                 ownershipType: String,
                 employmentLength: Int,
                 intent: String,
                 amountRequested: Double,
                 interestRate: Double,
                 priorDefault: Boolean,
                 creditHistory: Int,
                 grantorUsername: String,
                 status: String = "-",
                 loanGrade: String = "-"
               )
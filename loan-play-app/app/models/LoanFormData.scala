package models

case class LoanFormData(
  personAge: Int,
  personIncome: Int,
  ownershipType: String,
  employmentLength: Int,
  intent: String,
  amountRequested: Double,
  interestRate: Double,
  priorDefault: Boolean,
  creditHistory: Int
) 
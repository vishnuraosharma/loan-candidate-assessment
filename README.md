# Loan Application System

A Play Framework-based web application for loan management, featuring machine learning-powered loan grading and approval status prediction.

## Features

- **User Authentication**
  - Role-based access (Loan Grantor and Loan Requester)
  - Secure signup and login functionality

- **Loan Management**
  - Submit loan applications with detailed information
  - Automated loan grading system
  - Loan approval status prediction
  - Feature importance visualization

- **Analytics Dashboard**
  - Acceptance/rejection rates
  - Average loan amounts
  - Grade distribution charts
  - Purpose distribution analysis

## Stack

- **Backend**
  - Play Framework 3.0.5
  - Scala 2.13
  - Apache Spark ML 3.5.0
  - Machine Learning Models:
    - Random Forest Classifier (Loan Status Prediction)
    - Multilayer Perceptron Classifier (Loan Grading)

- **Frontend**
  - Twirl Template Engine
  - Custom CSS
  - JavaScript

## Prerequisites

- JDK 11
- sbt
- Apache Spark 3.5.0
- Scala 2.13.14

## Installation

1. Clone the repository
2. Navigate to the project directory
3. Run the application

```sbt run```

## Project Structure

loan-play-app/

├── app/

│ ├── controllers/ # Application controllers

│ ├── models/ # Data models

│ ├── services/ # Business logic and ML services

│ └── views/ # Twirl templates

├── conf/

│ ├── application.conf # Application configuration

│ └── routes # URL routing configuration

├── model/ # ML model artifacts

└── public/ # Static assets


## Machine Learning Pipeline

Data sourced from [Loan Approval Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e10/overview)

The application uses two main ML models:
1. Loan Status Model: Predicts approval/rejection
2. Loan Grader Model: Assigns loan grades (A-G)

Features considered include:
- Personal information (age, employment length)
- Financial details (income, loan amount)
- Credit history
- Property ownership status
- Loan purpose

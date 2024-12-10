name := """loan-play-app"""
organization := "com.vrszs"

version := "1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .enablePlugins(PlayScala)
  .settings(
    // Add JVM options for Spark compatibility
    javaOptions ++= Seq(
      "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.ref=ALL-UNNAMED",
      "--add-opens=java.base/java.math=ALL-UNNAMED",
      "--add-opens=java.base/java.text=ALL-UNNAMED",
      "--add-opens=java.base/java.time=ALL-UNNAMED",
      "--add-opens=java.base/java.util.stream=ALL-UNNAMED"
    ),
    fork := true
  )
scalaVersion := "2.13.14"

// Resolve dependency conflicts
ThisBuild / libraryDependencySchemes ++= Seq(
  "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always,
  "org.scala-lang.modules" %% "scala-parser-combinators" % VersionScheme.Always
)

libraryDependencies += guice
libraryDependencies += "org.scalatestplus.play" %% "scalatestplus-play" % "7.0.1" % Test

// Use consistent Spark version
val sparkVersion = "3.5.0"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion，
  "org.scalatest" %% "scalatest" % "3.2.15" % Test,
  "org.apache.spark" %% "spark-sql" % "3.3.0" % Test,
  "org.apache.spark" %% "spark-mllib" % "3.3.0" % Test
)

// Force resolution of conflicting dependencies
dependencyOverrides ++= Seq(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.15.2",
  "com.fasterxml.jackson.core" % "jackson-core" % "2.15.2",
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.15.2",
  "org.scala-lang.modules" %% "scala-parser-combinators" % "2.3.0"
)

// Adds additional packages into Twirl
//TwirlKeys.templateImports += "com.vrszs.controllers._"

// Adds additional packages into conf/routes
// play.sbt.routes.RoutesKeys.routesImport += "com.vrszs.binders._"

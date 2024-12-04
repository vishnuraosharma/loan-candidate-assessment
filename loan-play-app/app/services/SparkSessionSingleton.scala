package services

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import javax.inject.Singleton

@Singleton
object SparkSessionSingleton {
  lazy val spark: SparkSession = {
    // Set Hadoop home directory for Windows
    System.setProperty("hadoop.home.dir", "C:\\winutils\\")

    // Spark Configuration
    val conf = new SparkConf()
      .setAppName("PlaySparkApp")  // Set your Application name
      .setMaster("local[*]") // Run locally with as many working processors as logical cores
      .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer") // Set serializer to Java

    // Initialize SparkContext with the configuration
    val sc = new SparkContext(conf)

    // Create and return SparkSession
    SparkSession.builder().config(sc.getConf).getOrCreate()
  }
}
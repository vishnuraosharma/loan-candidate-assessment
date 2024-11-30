// Databricks notebook source
// MAGIC %md
// MAGIC # This is a Exploration on methods to visualize a Scala ML model
// MAGIC  The plan was to explore the hidden features and demonstrate the Users of the model a better understanding on the model. The test here is done through using a simpler model(Decision Tree, which is a simpler version fo RandomForest.) The goal is to explore different ways to acquire useful informations other than the prediction from the model.
// MAGIC ### The nature of Scala Model doesn't allow us to explore the model's reaction with single feature entry as we would have hoped in comparison to Python which comes equipped with extensive library for such operations.

// COMMAND ----------

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder
  .appName("Loan Grader")
  .config("spark.master", "local")
  .getOrCreate()

import spark.implicits._

// COMMAND ----------

// MAGIC %md
// MAGIC ### Load the model here, then we can extract valuable information from it.

// COMMAND ----------

import org.apache.spark.ml.classification.DecisionTreeClassificationModel

// Specify the same path where the model was saved
val modelPath = "/Users/momoga121@gmail.com/"

// Load the model
val loadedModel = DecisionTreeClassificationModel.load(modelPath)

println("Model loaded successfully!")
println(s"Loaded Model:\n${loadedModel.toDebugString}")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Keep in mind although the below explores the different aspects of our model, it is by no means the accuracy for a single or any prediction. But focus on the general model. I will list two methods, one implemented with the purpose of testing and displaying raw data, and one for the play framework. The ones with data will also need to come directly from your oww, so either download it and import or///

// COMMAND ----------

// DBTITLE 1,Feature Importance Evaluation (Play Framework)
def getFeatureImportance(): Action[AnyContent] = Action {
    val featureImportances = model.featureImportances.toArray
    val featureData = featureImportances.zipWithIndex.map { case (value, index) =>
        Json.obj("feature" -> s"Feature $index", "importance" -> value)
    }
    Ok(Json.toJson(featureData))
}

<div id="feature-importance-chart"></div>
<script>
fetch('/feature-importance')
    .then(response => response.json())
    .then(data => {
        const features = data.map(d => d.feature);
        const importances = data.map(d => d.importance);

        const trace = {
            x: features,
            y: importances,
            type: 'bar'
        };

        Plotly.newPlot('feature-importance-chart', [trace], {
            title: 'Feature Importance',
            xaxis: { title: 'Features' },
            yaxis: { title: 'Importance' }
        });
    });
</script>

// COMMAND ----------

// DBTITLE 1,Feature Importance Evaluation
val featureImportances = loadedModel.featureImportances.toArray
val featureData = featureImportances.zipWithIndex.map { case (value, index) =>
  Map("feature" -> s"Feature $index", "importance" -> value)
}
println(featureData)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Input data Visualization

// COMMAND ----------

// DBTITLE 1,Input Visualization with Play Framework
def getFeatureDistribution(): Action[AnyContent] = Action {
    val featureStats = data.describe().toJSON.collect()
    Ok(Json.toJson(featureStats))
}

<div id="data-distribution"></div>
<script>
fetch('/feature-distribution')
    .then(response => response.json())
    .then(data => {
        const features = ['Feature1', 'Feature2', 'Feature3']; // Replace with actual features
        const traces = features.map((feature, i) => ({
            x: data.map(d => d[feature]),
            type: 'histogram',
            name: feature
        }));

        Plotly.newPlot('data-distribution', traces, {
            title: 'Data Distribution',
            barmode: 'overlay'
        });
    });
</script>


// COMMAND ----------

// DBTITLE 1,Input Visualization
val featureStats = data.describe()

featureStats.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Label Distribution (This one is more specifically for the decision tree and loan Grade model, as there are multiple labels instead of the binary)

// COMMAND ----------

// DBTITLE 1,Label Distribution with Play Framework
def getClassDistribution(): Action[AnyContent] = Action {
    val classCounts = data.groupBy("label").count().toJSON.collect()
    Ok(Json.toJson(classCounts))
}

<div id="class-distribution"></div>
<script>
fetch('/class-distribution')
    .then(response => response.json())
    .then(data => {
        const labels = data.map(d => d.label);
        const counts = data.map(d => d.count);

        const trace = {
            labels: labels,
            values: counts,
            type: 'pie'
        };

        Plotly.newPlot('class-distribution', [trace], {
            title: 'Class Distribution'
        });
    });
</script>


// COMMAND ----------

// DBTITLE 1,Label Distribution
val classCounts = data.groupBy("indexedlabel").count()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Confusion Matrix
// MAGIC a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm.
// MAGIC Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa).
// MAGIC
// MAGIC The examples below also made the assumptions that you have your predictions ready
// MAGIC

// COMMAND ----------

// DBTITLE 1,Confusion  Matrix with Play Framework
def getConfusionMatrix(): Action[AnyContent] = Action {
    val rdd = predictions.select("prediction", "indexedLabel").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val metrics = new MulticlassMetrics(rdd)
    val confusionMatrix = metrics.confusionMatrix.toArray.grouped(metrics.labels.length).toArray
    Ok(Json.toJson(confusionMatrix))
}

<div id="confusion-matrix"></div>
<script>
fetch('/confusion-matrix')
    .then(response => response.json())
    .then(matrix => {
        const data = [{
            z: matrix,
            type: 'heatmap'
        }];

        Plotly.newPlot('confusion-matrix', data, {
            title: 'Confusion Matrix',
            xaxis: { title: 'Predicted Labels' },
            yaxis: { title: 'Actual Labels' }
        });
    });
</script>


// COMMAND ----------

// DBTITLE 1,Confusion Matrix
val rdd = predictions.select("prediction", "indexedLabel").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
val metrics = new MulticlassMetrics(rdd)
val confusionMatrix = metrics.confusionMatrix

println("Confusion matrix:")
println(confusionMatrix)

// COMMAND ----------

// MAGIC %md
// MAGIC ### ROC curve(For multi-class)/ (Can be modified easily for binary labels)
// MAGIC For demo purposes, the play framework is done for the single binary labels, while the scala code alone is for the multiclass.

// COMMAND ----------

// DBTITLE 1,With Play Framework
def getROCCurve(): Action[AnyContent] = Action {
    val binarySummary = model.evaluate(predictions).asInstanceOf[BinaryClassificationMetrics]
    val rocData = binarySummary.roc.collect()
    val rocPoints = rocData.map { case (fpr, tpr) => Json.obj("fpr" -> fpr, "tpr" -> tpr) }
    Ok(Json.toJson(rocPoints))
}

<div id="roc-curve"></div>
<script>
fetch('/roc-curve')
    .then(response => response.json())
    .then(data => {
        const fpr = data.map(d => d.fpr);
        const tpr = data.map(d => d.tpr);

        const trace = {
            x: fpr,
            y: tpr,
            type: 'scatter',
            mode: 'lines',
            name: 'ROC Curve'
        };

        Plotly.newPlot('roc-curve', [trace], {
            title: 'ROC Curve',
            xaxis: { title: 'False Positive Rate' },
            yaxis: { title: 'True Positive Rate' }
        });
    });
</script>


// COMMAND ----------

// Run the model on test dataset
val predictions = loadedModel.transform(dfTest)
val predictionAndLabels = predictions.select("prediction", "label")
  .rdd.map(row => (row.getAs[Double]("prediction"), row.getAs[DenseVector]("label").toArray(0)))
// Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
// ROC Curve
val roc = metrics.roc
roc.collect().foreach { case(fpr, tpr) =>
  println(s"False Positive Rate = $fpr, True Positive Rate = $tpr")
// Need to plot these but I didn't get to finish it

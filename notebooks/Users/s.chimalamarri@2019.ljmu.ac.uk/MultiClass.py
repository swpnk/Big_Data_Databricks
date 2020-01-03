# Databricks notebook source
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

# COMMAND ----------

data = spark.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema","true").load("FileStore/tables/sf_data/train.csv")

# COMMAND ----------

# MAGIC %Columns which are not Required
# MAGIC 
# MAGIC drop_list = ['Dates','DayOfWeek','Pd-District','Resolution','Address','X','Y']
# MAGIC data = data.select([column for column in data.columns if column not in drop_list])
# MAGIC data.show(5)

# COMMAND ----------

data.groupBy("Category").count().orderBy(col("count").desc()).show()

# COMMAND ----------

data.groupBy("Descript").count().orderBy(col("count").desc()).show()

# COMMAND ----------

regexTokenizer = RegexTokenizer(inputCol = "Descript", outputCol = "words", pattern = "\\W")
add_stopwords = ["http","https","amp","rt","t","c","the"]
stopwordsRemover = StopWordsRemover(inputCol = "words", outputCol = "filtered").setStopWords(add_stopwords)
countVectors = CountVectorizer(inputCol = "filtered", outputCol = "features", vocabSize = 10000, minDF =5)
label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")

# COMMAND ----------

pipeline = Pipeline(stages = [regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7,0.3], seed = 100)
print("Training Data"+str(trainingData.count()))
print("TestData"+str(testData.count()))

# COMMAND ----------

lr = LogisticRegression(maxIter = 20, regParam = 0.3, elasticNetParam=0)
lrmodel = lr.fit(trainingData)
predictions = lrmodel.transform(testData)

predictions.filter(predictions['prediction'] == 0).select("Descript","Category","probability","label","prediction").orderBy("probability", ascending = False).show(n=10,truncate=30)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(predictionCol = "prediction")
evaluator.evaluate(predictions)

# COMMAND ----------

paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.1,0.3,0.5]).addGrid(lr.elasticNetParam, [0.0,0.1,0.2]).addGrid(lr.maxIter, [10,20,50]).build())
cv = CrossValidator(estimator = lr, estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=5)

cvmodel = cv.fit(trainingData)

# COMMAND ----------

predictions_cv= cvmodel.transform(testData)
evaluator.evaluate(predictions_cv)
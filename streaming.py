from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, explode, array, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression,OneVsRest
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import isnan, when, count, col, lit, sum
from pyspark.sql.functions import (to_date, datediff, date_format,month)
from pyspark.sql.functions import col, explode, array, lit
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType, MapType
from pyspark.sql.functions import to_timestamp
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

#Iniciar entrenamiento
def init_training(df):

    #Imprime la cantidad de registros y atributos respectivamente
    print("Registros Iniciales:",df.count(),", Atributos Iniciales:",len(df.columns))
    df.show(100)
    #Tipo de los atributos
    print("first print schema")
    df.printSchema()
    columnasEliminar = ['_c0', 'ID', 'Case Number', 'Primary Type', 'Description', 'Ward', 'Latitude', 'Longitude', 'Location', 'Updated On','Year']

    df = df.select([column for column in df.columns if column not in columnasEliminar])


    #elimanar filas con valores nulos 
    columnasNulos = ['District', 'Community Area', 'X Coordinate', 'Y Coordinate', 'Location Description']
    df = df.dropna(how='any', subset=columnasNulos)
    #serializar columnas
    columnasSerializar = ['IUCR', 'Location Description', 'FBI Code', 'Block']
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in columnasSerializar]
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    df = df.select([column for column in df.columns if column not in columnasSerializar])
    #conervirt booleanos a enteros
    df = df.withColumn("Arrest", when(col("Arrest"), lit(1)).otherwise(0)).withColumn("Domestic", when(col("Domestic"), lit(1) ).otherwise(0))
    
    #correlacion y balanceo
    # borrar la corr > 0.9
    data = df.toPandas()
    corr = data.corr()
    print("CORRELACIÓN============")
    print(corr)
    df = df.drop("Beat")
    # distribucion de las clases
    print("DISTRIBUCION CLASES============")
    df.groupBy('Arrest').count().show()

    #balanceo
    major_df = df.filter(col("Arrest") == 0)
    minor_df = df.filter(col("Arrest") == 1)
    ratio = int(major_df.count()/minor_df.count())
    print("ratio: {}".format(ratio))

    a = range(ratio+1)
    # duplicate the minority rows
    oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
    # combine both oversampled minority rows and previous majority rows 
    combined_df = major_df.unionAll(oversampled_df)
    combined_df.groupBy('Arrest').count().show()

    
    #entrenamiento
    cols=combined_df.columns
    cols.remove("Arrest")
    cols.remove("Block_index")
    # df=df.drop(cols[15])
    # df=df.drop(cols[12])
    # df=df.drop(cols[13])
    # df.printSchema()
    # cols=df.columns
    vector = VectorAssembler(inputCols=cols,outputCol="features")
    data = vector.transform(combined_df)
    data = data.select('features', 'Arrest')
    data = data.selectExpr("features as features", "Arrest as label")


    #division del dataset 70% entrenamiento - 30% pruebas
    train, test = data.randomSplit([0.6, 0.4], seed=20)
    #instacia del evaluador
    evaluator = BinaryClassificationEvaluator()
    print("training dataset")
    print(train.features.show)
    
    #regresion logistica
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=2)
    lrModel = lr.fit(train)
    predictions = lrModel.transform(test)
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
    print("=========================================================================================================================")
    print("REGRESION LOGISTICA")
    print('area bajo el ROC', evaluator.evaluate(predictions))
    print("presicion de la regresion logistica: ", accuracy)
    print("=========================================================================================================================")
    
    #arboles de decision
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3,maxBins=35000)
    dtModel = dt.fit(train)
    predictionsDt = dtModel.transform(test)
    accuracy2 = predictionsDt.filter(predictionsDt.label == predictionsDt.prediction).count() / float(predictionsDt.count())
    print("=========================================================================================================================")
    print("ARBOLES DE DECISION")
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictionsDt, {evaluator.metricName: "areaUnderROC"})))
    print("presicion de los arboles de decision: ", accuracy2)
    print("=========================================================================================================================")

    #random forest
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3,maxBins=35000)
    rfModel = rf.fit(train)
    predictionsRf = rfModel.transform(test)
    accuracy3 = predictionsRf.filter(predictionsRf.label == predictionsRf.prediction).count() / float(predictionsRf.count())
    print("=========================================================================================================================")
    print("RANDOM FOREST") 
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictionsRf, {evaluator.metricName: "areaUnderROC"}))) 
    print("presicion random forest: ", accuracy3)
    print("=========================================================================================================================")

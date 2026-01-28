
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:36:02 2022

@author: geethumuru
"""



from __future__ import print_function

import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
import time

from pyspark.mllib.evaluation import MulticlassMetrics


from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import PCA
from pyspark.ml.regression import LinearRegression

from pyspark.sql.functions import col

from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.classification import DecisionTreeClassifier
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from pyspark.ml.classification import RandomForestClassifier


DataFile="/Users/geethumuru/Documents/Grad School/BU/CS777/Homeworks/Final Project/Aus_Weather_data_clean_final.csv"



sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()


# Read data file
rdd_datafile = sc.textFile(DataFile)

header = rdd_datafile.first()
rdd_datafile = rdd_datafile.filter(lambda line: line != header)


# Split data by comma
rdd_d = rdd_datafile.map(lambda x: x.split(','))


# Change to dataframe
df = rdd_d.toDF()



df = df.withColumnRenamed("_1","index")
df = df.withColumnRenamed("_2","WindGust")
df = df.withColumnRenamed("_3","Cloud")
df = df.withColumnRenamed("_4","Pressure")
df = df.withColumnRenamed("_5","Humidity")
df = df.withColumnRenamed("_6","Temp")
df = df.withColumnRenamed("_7","RToday")
df = df.withColumnRenamed("_8","RTomorrow")



df = df.select(*(col(c).cast("double").alias(c) for c in df.columns)).cache()


# make sure none of the column values are null
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()



fscore_Today = []
fscore_Tomorrow = []
fscore_order = ["All",'WindGust', 'Cloud', 'Pressure', 'Humidity', 'Temp']

######### Logistic Regression With all Features Together For Rain Today#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['WindGust', 'Cloud', 'Pressure', 'Humidity', 'Temp'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RToday')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


log_reg = LogisticRegression(featuresCol='features',labelCol='RToday')

fit_model = log_reg.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RToday"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
cm = metrics.confusionMatrix().toArray().astype(int)  
    
print("All Features to Predict Rain Today Linear Regression")
print("F Score: ")
print(f_score)
print("")
print("Confusion Matrix: ")
print(cm)
fscore_Today.append(f_score)



######### Logistic Regression With all Features Together For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['WindGust', 'Cloud', 'Pressure', 'Humidity', 'Temp'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


log_reg_tomorrow = LogisticRegression(featuresCol='features',labelCol='RTomorrow')

fit_model = log_reg_tomorrow.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
cm = metrics.confusionMatrix().toArray().astype(int)  
    
print("All Features to Predict Rain Tomorrow Linear Regression")
print("F Score: ")
print(f_score)
print("")
print("Confusion Matrix: ")
print(cm)
fscore_Tomorrow.append(f_score)

#selection = ChiSqSelector(numTopFeatures=200, featuresCol=idf.getInputCol(), outputCol="features",labelCol = "label")




######### Logistic Regression With WindGust  For Rain Today#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['WindGust'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RToday')

# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RToday"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("WindGust to Predict Rain Today Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Today.append(f_score)


######### Logistic Regression With Cloud  For Rain Today#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Cloud'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RToday')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RToday"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
 
    
print("Cloud to Predict Rain Today Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Today.append(f_score)


######### Logistic Regression With Pressure  For Rain Today#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Pressure'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RToday')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RToday"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)

    
print("Pressure to Predict Rain Today Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Today.append(f_score)


######### Logistic Regression With Humidity  For Rain Today#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Humidity'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RToday')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RToday"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    

    
print("Humidity to Predict Rain Today Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Today.append(f_score)


######### Logistic Regression With Temp For Rain Today#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Temp'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RToday')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RToday"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("Temperature to Predict Rain Today Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Today.append(f_score)




######### Logistic Regression With WindGust  For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['WindGust'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg_tomorrow.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("WindGust to Predict Rain Tomorrow Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Tomorrow.append(f_score)


######### Logistic Regression With Cloud For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Cloud'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg_tomorrow.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("WindGust to Predict Rain Tomorrow Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Tomorrow.append(f_score)



######### Logistic Regression With Pressure For Rain Tomorrow#######
# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Pressure'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg_tomorrow.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("Pressure to Predict Rain Tomorrow Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Tomorrow.append(f_score)


######### Logistic Regression With Humidity For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Humidity'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg_tomorrow.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("Humidity to Predict Rain Tomorrow Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Tomorrow.append(f_score)

######### Logistic Regression With Temperature For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Temp'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)


fit_model = log_reg_tomorrow.fit(df_train)
results = fit_model.transform(df_test)

l_p_Results_total = results.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("Temperature to Predict Rain Tomorrow Linear Regression")
print("F Score: ")
print(f_score)
print("")

fscore_Tomorrow.append(f_score)




# Plot to determine which feature is the best
plt.plot(fscore_order, fscore_Today, label = "Rain Today")
plt.plot(fscore_order, fscore_Tomorrow, label = "Rain Tomorrow")
plt.ylabel("fscore")
plt.title("Fscore for each feature")
plt.legend()
plt.show()






# Saving f scores for all the classifiers
fscore_dt = []
fscore_rfc = []
fscore_lr = []
#fscore_order = ['Humidity', 'All']
fscore_dt.append("Decision Tree")
fscore_rfc.append("Random Forest")

fscore_lr.append("Logistic Regression")
fscore_lr.append(fscore_Tomorrow[4])
fscore_lr.append(fscore_Tomorrow[0])



# since the two that are the highest are humidity and all the features 
# together, both will be analyzed with two more classifiers

######### Decision Tree With Humidity For Rain Tomorrow#######


# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Humidity'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)

decisionT = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'RTomorrow', maxDepth = 3)
dtModel = decisionT.fit(df_train)
predictions = dtModel.transform(df_test)

l_p_Results_total = predictions.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("Humidity to Predict Rain Tomorrow with Decision Tree")
print("F Score: ")
print(f_score)
print("")
fscore_dt.append(f_score)

######### Decision Tree With All Features For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['WindGust', 'Cloud', 'Pressure', 'Humidity', 'Temp'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)

dtModel = decisionT.fit(df_train)
predictions = dtModel.transform(df_test)

l_p_Results_total = predictions.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    
    
print("All Features to Predict Rain Tomorrow with Decision Tree")
print("F Score: ")
print(f_score)
print("")
fscore_dt.append(f_score)






######### Decision Tree With Humidity For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['Humidity'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)

rfc = RandomForestClassifier(featuresCol = 'features', labelCol = 'RTomorrow')
dtModel = rfc.fit(df_train)
predictions = dtModel.transform(df_test)

l_p_Results_total = predictions.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    

print("Humidity to Predict Rain Tomorrow with Random Forest Classifier")
print("F Score: ")
print(f_score)
print("")
fscore_rfc.append(f_score)

######### Decision Tree With All Features For Rain Tomorrow#######

# Use VectorAssembler to combine the values of the independent valriables into one vector
vectorAssembler = VectorAssembler(inputCols = ['WindGust', 'Cloud', 'Pressure', 'Humidity', 'Temp'], outputCol = 'features')

features_df = vectorAssembler.transform(df)
features_df_today = features_df.select('features', 'RTomorrow')


# split data into two
df_train, df_test = features_df_today.randomSplit(weights = [0.8,0.2],seed = 200)

rfc = RandomForestClassifier(featuresCol = 'features', labelCol = 'RTomorrow')
dtModel = rfc.fit(df_train)
predictions = dtModel.transform(df_test)

l_p_Results_total = predictions.select(["prediction","RTomorrow"]).rdd.map(lambda x: (float(x[0]),float(x[1])))


metrics = MulticlassMetrics(l_p_Results_total)
f_score = metrics.fMeasure(0.0)
    

print("All Features to Predict Rain Tomorrow with Random Forest Classifier")
print("F Score: ")
print(f_score)
print("")
fscore_rfc.append(f_score)


# Specify the Column Names while initializing the Table
myTable = PrettyTable(["Model", "Fscore Humidity", "Fscore All Features"])
 
myTable.add_row(fscore_lr)
myTable.add_row(fscore_dt)
myTable.add_row(fscore_rfc)

print(myTable)



  
sc.stop()
spark.stop()





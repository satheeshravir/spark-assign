import sys
import os
SPARK_HOME = "/home/maxsteal/Downloads/spark-1.5.2-bin-hadoop2.6/" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """ 
    feats = line.strip().split("\t") 
    # labels must be at the beginning for LRSGD, it's in the end in our data, so 
    # putting it in the right place
    label = feats[0].strip()
    feats = [feats[1].strip()]
    feats.insert(0,label)
    #features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, feats)

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("train/study_hours.txt")

parsedData = data.map(mapper)

print "########################################"

# Train model
model = LogisticRegressionWithSGD.train(parsedData)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point.label), 
        model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))

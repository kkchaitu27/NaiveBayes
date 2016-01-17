"""
This module contains functions for Naive Bayes classificiation algorithm.
"""

from collections import Counter
import pandas as pd
import numpy as np
import operator

def get_attribute_values(df,attribute):
    """
    Input : df-Pandas Data Frame and attribute-attribute for which unique values have to be computed
    Returns the unique values of an attribute in the data frame
    """
    return df[attribute].unique()

#MAP estimate of NaiveBayes Algortihm
def NaiveBayesMAP(df,target,betak,betam):
   """
    Input : df:Pandas DataFrame of original data and target: target attribute
    Returns prior probability for each word in the above list
   """
   attributes = df.columns.values
   classes = np.sort(get_attribute_values(df,target))
   
   betak = pd.Series(betak,index=classes)
   betam = pd.Series(betam,index=classes)

   class_frequency = df[target].value_counts().sort_index()
   class_probability = 1.0*(class_frequency+(betak-1))/(sum(class_frequency)+ sum(betam-1))
   class_probability_dict = dict(zip(class_probability.index,class_probability))
   attribute_probability_dict = {}   

   for attribute in attributes:
        attribute_values = get_attribute_values(df,attribute)
        for value in attribute_values:
           i = 1 
           for clazz in classes:
              qury = target + "==" + str(clazz) + "and " + str(attribute) + "==" + str(value)
              key = str(attribute) + "_" + str(value) + "_" + str(clazz)
              attribute_probability_dict[key] = 1.0 * (len(df.query(qury)) + betak[i]-1)/(len(df[df[target]==clazz])+sum(betam-1))
              i = i+1

   return list([attribute_probability_dict,class_probability_dict])

def get_attribute_probability(row,attributes,attribute_probability_dict,clazz):
    """
    This function returns a  attribute level probability of a given record for articular class
    """
    probability = 0    

    for attribute in attributes:
         key = attribute + "_" + str(row[attribute]) + "_" + str(clazz)
         if key in attribute_probability_dict.keys():
              probability = probability + np.log(attribute_probability_dict[key])
         else:
              print key
              print "Key doesnot exist in model"
              sys.exit(0)

    return probability



def get_prediction(row,model,attributes):
    """
    This function returns a  classification for the given record.
    """
    class_probability_dict = model[1]
    attribute_probability_dict = model[0]
    best_class = max(class_probability_dict.iteritems(), key=operator.itemgetter(1))[0]
    best_probability = np.log(class_probability_dict[best_class])+get_attribute_probability(row,attributes,attribute_probability_dict,best_class)
    for clazz in class_probability_dict.keys():
       clazz_probability = np.log(class_probability_dict[clazz])+get_attribute_probability(row,attributes,attribute_probability_dict,clazz)
       if clazz_probability > best_probability:
          best_class = clazz
          best_probability = clazz_probability
    return best_class


def predict(model,predData):
   """
    Input : model:Model returned by NaiveBayesMap Method and predData:Pandas DataFrame of original data
    Returns predictions for predData
   """
   attributes = predData.columns.values
   predictions = []
   for index, row in predData.iterrows():
       predictions.append(get_prediction(row, model,attributes))

   return predictions



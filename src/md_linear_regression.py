#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:02:50 2020

@author: zakk
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:57:15 2020

@author: zakk
"""

import numpy as np
import pandas as pd
import random
from time import time

# Call to run
def run():
    ms1 = int(time() * 1000)
    multi_reg(100, 0.001)
    ms2 = int(time() * 1000)
    difference = ms2 - ms1
    print
    print "Time passed:", difference, "ms"

# Function to predict the PM2.5 value
# Following the formula for linear regressions:
# Y = a0x0 + a1x1 + a2x2 + ... + anxn + b
#
# Takes the coefficients and multiplies them with the data values
# in the dataset using NumPy's multiply() function. Then these
# values are summed up, b is added, and the final prediction for
# the entire row is saved in a list. The list is returned once
# all predictions have been made.
def predict(data, targets, coefficients, b):
    # initialize list
    predictions = []
    # loop through data
    for i in range(len(data)):
        mult = np.multiply(data[i], coefficients)
        added = np.sum(mult)
        added += b
        predictions.append(added)   # add to list
    return predictions

# Function to calculate the cost of a given hypothesis
# Cost function = RMSE equation
#
# Calculates the RMSE for a given hypothesis and returns the value
# as the associated cost of the hypothesis.
def calc_cost(predictions, targets):
    # initialize
    summation = 0.0
    # loops through predictions
    for i in range(len(predictions)):
        curr_margin = predictions[i] - targets[i]
        curr_margin = curr_margin**2
        summation += curr_margin
    summation = summation/len(targets)
    rmse = np.sqrt(summation)
    return rmse

# Function to adjust the coefficients
# Using partial derivatives of the cost function
#
# Calculates the value by which each coefficient is adjusted. To ensure
# we get different sets of coefficients, the parameter pos_neg will
# will slightly alter the adjustment.
def adjust(predictions, targets, learning_rate, coefficients, pos_neg):
    summation = 0.0
    rand = random.random()
    ret_coeff = []
    for i in range(len(predictions)):
        summation += ((predictions[i] - float(targets[i][0]))**2)
    adj_val = learning_rate*(summation/float(len(targets)*(2+(pos_neg*rand))))
    rand = random.random()
    for i in range(len(coefficients)):
        ret_coeff.append(coefficients[i] - (learning_rate*adj_val*(rand)))
    return ret_coeff

# Function to run the multidimensional linear regression
#
# Begins by loading the data as a Pandas DataFrame. Data cleaning:
#   # 1. Since we are predicting the PM2.5 value, we don't want any rows
#   # in the data that are missing the PM2.5 value => removed all rows with
#   # null in the PM2.5 column.
#   # 2. Found that the year column wasn't contributing to the overall accuracy
#   # and was instead worsening the accuracy => removed the column 'year'
#   # 3. The values in the column 'cbwd' are not numerical => removed the column 'cbwd'
# Variables are initialized and the loop runs for n iterations. In each iteration,
# the predictions are made, the costs are calculated and compared, the optimal
# coefficients are saved, and the coefficients are adjusted for the next loop.
# Prints statements with each iteration stating the generation number, the two RMSEs,
# and which set of coefficients is being chosen as optimal.
def multi_reg(n_iter, learning_rate):
    # loading the data
    data = pd.DataFrame(pd.read_csv('Desktop/CS 456/PRSA_data_2010.1.1-2014.12.31.csv', index_col=0))
    data = data.dropna()    # cleaning
    data = data.drop(columns=['pm2.5','cbwd','year'])   # cleaning
    data = np.array(data)   # converted to an array for easier access
    
    # loading the target data
    targets = pd.DataFrame(pd.read_csv('Desktop/CS 456/PRSA_data_2010.1.1-2014.12.31.csv', index_col=0))
    targets = targets.dropna()  # cleaning
    targets = targets.drop(columns=['year','month','day','hour','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir'])   # cleaning
    targets = np.array(targets) # converted to an array for easier access
    
    # randomly initializing coefficients
    coefficients = np.random.rand(1,len(data[0]))
    b = random.random()
    predictions = predict(data, targets, coefficients, b)
    
    # placeholder variables
    best_rmse = float(calc_cost(predictions, targets))
    best_coeff = coefficients
    best_pred = predictions
    
    print "Starting RMSE:", best_rmse
    
    # first adjustment
    coefficients = adjust(predictions, targets, learning_rate, coefficients, 0)
    
    # loops for n_iter iterations
    for i in range(n_iter):
        print
        print "---------------------------------------------------------------"
        print "Generation", (i+1), ":"
        
        # making the predictions
        predictions = predict(data, targets, coefficients, b)
        
        # calculating the costs
        rmse = float(calc_cost(predictions, targets))
        
        print "Resulting RMSEs:"
        print "- Current best:", best_rmse
        print "- Using current set of coefficients:", rmse
        
        # choosing the optimal coefficients
        if rmse < best_rmse:
            best_rmse = rmse
            best_coeff = coefficients
            best_pred = predictions
            b = b - learning_rate*best_rmse
            print "--> Choosing current set of coefficients."
            if (i+1) != n_iter:
                coefficients = adjust(best_pred, targets, learning_rate, best_coeff, 0)
        else:
            coefficients = adjust(best_pred, targets, learning_rate, best_coeff, 1)
            print "--> Keeping older set of coefficients."
    print
    print "--FINISH--"
    print
    print "Across", n_iter, "generations, the best RMSE was", best_rmse, "with the following coefficients:"
    print best_coeff
    print "The value for b was", b, "and the learning rate was", learning_rate
        
    
run()

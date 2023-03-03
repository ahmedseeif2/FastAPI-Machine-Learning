#!/usr/bin/env python
# coding: utf-8

#Machine Learning Task Assesmet for Sourcefuse
#Ahmed Ceifelnasr
#ahmedseeif2@gmail.com

# # Task 
#1. Using the Iris dataset, train models to solve the following: KNN, Random Forest Classifier, SVM classifier, and a logistic regression classifier
#2. Using FastApi, create an endpoint for each model that will allow you to pass in a target to each endpoint and execute the model
#3. Push the code to GitHub  

#1. Using the Iris dataset, train models to solve the following: KNN, Random Forest Classifier, SVM classifier, and a logistic regression classifier




#Import modules
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Loading Iris Dataset
iris = load_iris()

# The Iris dataset is pretty much clean si I will not start data preprocessing or data visualization

# Getting features and targets from the dataset
X = iris.data
Y = iris.target

# Fitting KNN Model on the dataset
clf_KNN = KNeighborsClassifier()
clf_KNN.fit(X,Y)
 

# Fitting RandomForestClassifier Model on the dataset
clf_RF = RandomForestClassifier()
clf_RF.fit(X,Y)

# Fitting SVM Model on the dataset
clf_SVM = SVC()
clf_SVM.fit(X,Y)

# Fitting LogisticRegression Model on the dataset
clf_LG = LogisticRegression()
clf_LG.fit(X,Y)

# ### 2. Using FastApi, create an endpoint for each model that will allow you to pass in a target to each endpoint and execute the model

# Importing  modules
from fastapi import FastAPI
import uvicorn
 
# Declaring our FastAPI instance
app = FastAPI()

#To define our request body we’ll use BaseModel in pydantic module
from pydantic import BaseModel
#To define our request body we’ll create a class that inherits BaseModel and define the features as the attributes of that class
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
   
#The Endpoint: now that we have a request body all that’s left to do is to add an endpoint that’ll predict the class and return it as a response

#Endpoint for KNN
@app.post('/predict_KNN')
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf_KNN.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}

#Endpoint for Random Forest
@app.post('/predict_RandomForestClassifier')
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf_RF.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}

#Endpoint for SVM
@app.post('/predict_SVM')
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf_SVM.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}

#Endpoint for Logistic Regression
@app.post('/predict_LogisticRegression')
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf_LG.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}

# Thanks

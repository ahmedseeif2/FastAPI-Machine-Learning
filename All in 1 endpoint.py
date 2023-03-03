#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Creating an instance of the FastAPI class
app = FastAPI()

# Loading the iris dataset
iris = load_iris()

# Creating a Pydantic model for the input data
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Creating instances of the SVM, Random Forest, Logistic Regression, and KNN classifiers
svm = SVC()
rf = RandomForestClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier()

# Fitting the entire dataset to each classifier
svm.fit(iris.data, iris.target)
rf.fit(iris.data, iris.target)
lr.fit(iris.data, iris.target)
knn.fit(iris.data, iris.target)

# Creating an endpoint for the classification models
@app.post("/predict")
def predict(input_data: InputData):
    # Converting input data to a numpy array
    input_array = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    
    # Predicting the target value for the input data using each classifier
    svm_output = svm.predict(input_array)[0]
    rf_output = rf.predict(input_array)[0]
    lr_output = lr.predict(input_array)[0]
    knn_output = knn.predict(input_array)[0]

    # Converting the target values to strings
    svm_output_str = iris.target_names[svm_output]
    rf_output_str = iris.target_names[rf_output]
    lr_output_str = iris.target_names[lr_output]
    knn_output_str = iris.target_names[knn_output]

    # Returning the predicted target values as a JSON response
    return {"svm_prediction": svm_output_str, "rf_prediction": rf_output_str, "lr_prediction": lr_output_str, "knn_prediction": knn_output_str}


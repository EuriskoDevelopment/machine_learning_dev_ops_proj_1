# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The following project demonstrates how to move code from a data scientists notebook to production ready python code.
The first python file `churn_library.py` contains a library to predict costumer churn and has been written with coding best practices.
The second python file `churn_script_logging_and_tests.py` contains tests and logging needed to test the 
Your project description here.

The move from a Data Scientist to a Machine Learning Engineer requires a move to coding best practices. In this project, we are tasked with moving code from a notebook that completes the data science process, but doesn't lend itself easy to reproduce, production-level code, to two scripts:

1. The first script `churn_library.py` is a python library containing functions needed to complete the same data science process.

2. The second script `churn_script_logging_and_tests.py` contains tests and logging that test the functions of your library and log any errors that occur.  

The original python notebook `churn_notebook.ipynb` contains the code to be refactored.


## Running Files
To run the customer churn library run:
```console
ipython churn_library.py
```
This will create one logistic regression model and one random forest model in the /models/
folder.

Running the following code will test the customer churn library and will create a log:
```console
ipython churn_script_logging_and_tests.py
```

Except for the preset workspace the following packages and specific versions needs to be
installed with the following commands:
```console
pip install shap
python -m pip install scikit-learn==0.22
```

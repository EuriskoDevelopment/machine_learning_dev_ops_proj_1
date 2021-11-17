'''
This script will predict customer churn
'''

# import libraries
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

TRAIN_RESULT_SAVE_FOLDER = './images/results/'
EDA_RESULT_SAVE_FOLDER = './images/eda/'

category_list = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        data_frame = pd.read_csv(pth)
        # Add churn information to data set
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return data_frame
    except FileNotFoundError as error:
        raise error


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    try:
        assert data_frame.isnull().sum().sum() == 0

        fig = plt.figure(figsize=(20, 10))
        data_frame['Churn'].hist()
        fig.savefig(EDA_RESULT_SAVE_FOLDER + 'churn_hist.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        data_frame['Customer_Age'].hist()
        fig.savefig(EDA_RESULT_SAVE_FOLDER + 'customer_age.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
        fig.savefig(EDA_RESULT_SAVE_FOLDER + 'marital_status.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        sns.distplot(data_frame['Total_Trans_Ct'])
        fig.savefig(EDA_RESULT_SAVE_FOLDER + 'marital_status.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        fig.savefig(EDA_RESULT_SAVE_FOLDER + 'heat_map.png')
        plt.close(fig)
    except KeyError as err:
        raise err
    except AssertionError as err:
        raise err


def encoder_helper(data_frame, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    y = data_frame[response]
    X = pd.DataFrame()

    # gender encoded column
    gender_lst = []
    gender_groups = data_frame.groupby('Gender').mean()[response]

    for val in data_frame['Gender']:
        gender_lst.append(gender_groups.loc[val])

    data_frame['Gender_Churn'] = gender_lst

    # education encoded column
    edu_lst = []
    edu_groups = data_frame.groupby('Education_Level').mean()[response]

    for val in data_frame['Education_Level']:
        edu_lst.append(edu_groups.loc[val])

    data_frame['Education_Level_Churn'] = edu_lst

    # marital encoded column
    marital_lst = []
    marital_groups = data_frame.groupby('Marital_Status').mean()[response]

    for val in data_frame['Marital_Status']:
        marital_lst.append(marital_groups.loc[val])

    data_frame['Marital_Status_Churn'] = marital_lst

    # income encoded column
    income_lst = []
    income_groups = data_frame.groupby('Income_Category').mean()[response]

    for val in data_frame['Income_Category']:
        income_lst.append(income_groups.loc[val])

    data_frame['Income_Category_Churn'] = income_lst

    # card encoded column
    card_lst = []
    card_groups = data_frame.groupby('Card_Category').mean()[response]

    for val in data_frame['Card_Category']:
        card_lst.append(card_groups.loc[val])

    data_frame['Card_Category_Churn'] = card_lst

    X[category_lst] = data_frame[category_lst]
    return X


def perform_feature_engineering(data_frame, response='Churn'):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = data_frame[response]
    X = encoder_helper(data_frame, category_list, response)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(TRAIN_RESULT_SAVE_FOLDER + 'classification_report_RF.png')

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(TRAIN_RESULT_SAVE_FOLDER + 'classification_report_Log_reg.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    fig.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0
    assert x_train.shape[1] > 0
    assert x_test.shape[1] > 0
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # print("grid search")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    #print("log reg fit")
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    #print("save roc figures")
    # save roc figures
    fig = plt.figure()
    axis = plt.gca()
    lrc_plot = plot_roc_curve(lrc, x_test, y_test, ax=axis)
    fig.savefig(TRAIN_RESULT_SAVE_FOLDER + 'lin_reg_roc_curve.png')
    # plt.close(fig)

    # plots
    fig = plt.figure(figsize=(15, 8))
    axis = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    fig.savefig(TRAIN_RESULT_SAVE_FOLDER + 'comp_linreg_rf_roc_curve.png')
    plt.close(fig)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # and verify saved models by loading them
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    fig = plt.figure()
    axis = plt.gca()
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test, ax=axis)
    fig.savefig(TRAIN_RESULT_SAVE_FOLDER + 'lin_reg_roc_curve_saved_model.png')
    plt.close(fig)

    fig = plt.figure(figsize=(15, 8))
    axis = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    fig.savefig(TRAIN_RESULT_SAVE_FOLDER + 'comp_linreg_rf_roc_curve_saved_model.png')
    plt.close(fig)
    
    # SHAP library not working
    #print("shap")
    #fig = plt.figure(figsize=(15, 8))
    #explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    #shap_values = explainer.shap_values(x_test)
    #shap.summary_plot(shap_values, x_test, plot_type="bar")
    #fig.savefig(TRAIN_RESULT_SAVE_FOLDER + 'shap_best_rf_model.png')
    #plt.close(fig)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == "__main__":
    # import the data
    data = import_data("./data/bank_data.csv")

    # perform eda
    perform_eda(data)

    # split into training and testing data
    x_training, x_testing, y_training, y_testing = perform_feature_engineering(
        data, 'Churn')

    # train models and store results
    train_models(x_training, x_testing, y_training, y_testing)

    X = encoder_helper(data, category_list)
    rfc_model = joblib.load('./models/rfc_model.pkl')
    feature_importance_plot(rfc_model,
                            X,
                            TRAIN_RESULT_SAVE_FOLDER + './feature_importance_plot.png')

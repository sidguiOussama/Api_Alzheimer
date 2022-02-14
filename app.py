from flask import Flask,render_template,request,jsonify
import json
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.exceptions import ConvergenceWarning 
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc

import time
tps_tot = time.perf_counter()
acc = [] # list to store all performance metric
from sklearn import svm

app = Flask(__name__)


@app.route('/')
def index():
    return "hello"

@app.route('/api/alzheimer/oasis/longitudinal/dataset/display')
def displayDataSet():
    return jsonify(data)

@app.route('/api/alzheimer/oasis/dataset/describe')
def displayDataSetDescribe():
    info = df.describe()
    return jsonify(json.loads(info.to_json (orient='columns')))


@app.route('/api/alzheimer/oasis/dataset/clean')
def displayDataClean():
    global df
    # Nettoyage du dataset
    df = df.loc[df['Visit']==1] 
    # use first visit data only because of the analysis 
    df = df.reset_index(drop=True) 
    # reset index after filtering first visit data 
    df['M/F'] = df['M/F'].replace(['F','M'], [0,1]) # M/F column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented']) # Target variabl 
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) # Target 
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1) # Drop unnecessary columns
    return jsonify({'message': 'successful operation'})

@app.route('/api/alzheimer/oasis/dataset/delete/missingValues')
def deleteMissingValues():
    global df
    pd.isnull(df).sum() 
    # The column, SES has 8 missing values

    # Dropped the 8 rows with missing values in the column, SES
    df_dropna = df.dropna(axis=0, how='any')
    pd.isnull(df_dropna).sum()
    return jsonify({'message': 'successful operation'})

@app.route('/api/alzheimer/oasis/dataset/imputation')
def imputation():
    global df
    x = df['EDUC']
    y = df['SES']

    ses_not_null_index = y[~y.isnull()].index
    x = x[ses_not_null_index]
    y = y[ses_not_null_index]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    df.groupby(['EDUC'])['SES'].median()

    df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
    pd.isnull(df['SES']).value_counts()
    return jsonify({'message': 'successful operation'})

@app.route('/api/alzheimer/oasis/dataset/Splitting')
def Splitting():
    # Dataset with imputation
    Y = df['Group'].values # Target for the model
    X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']] # Features we use

    # splitting into three sets
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, random_state=0)

    # Feature scaling
    scaler = MinMaxScaler().fit(X_trainval)
    X_trainval_scaled = scaler.transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    # Dataset after dropping missing value rows
    Y = df_dropna['Group'].values # Target for the model
    X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']] # Features we use

    # splitting into three sets
    X_trainval_dna, X_test_dna, Y_trainval_dna, Y_test_dna = train_test_split(X, Y, random_state=0)

    # Feature scaling
    scaler = MinMaxScaler().fit(X_trainval_dna)
    X_trainval_scaled_dna = scaler.transform(X_trainval_dna)
    X_test_scaled_dna = scaler.transform(X_test_dna)
    return jsonify({'message': 'successful operation'})


@app.route('/api/alzheimer/oasis/dataset/LogisticRegression')
def logisticRegression():
    tps = time.perf_counter()

    # Dataset with imputation
    best_score=0
    kfolds=5 # set the number of folds

    for c in [0.001, 0.1, 1, 10, 100]:
        logRegModel = LogisticRegression(C=c)
        # perform cross-validation
        
        # Get recall for each parameter setting
        scores = cross_val_score(logRegModel, X_trainval, Y_trainval, cv=kfolds, scoring='accuracy')
        
        # compute mean cross-validation accuracy
        score = np.mean(scores)
        
        # Find the best parameters and score
        if score > best_score:
            best_score = score
            best_parameters = c

    # rebuild a model on the combined training and validation set
    SelectedLogRegModel = LogisticRegression(C=best_parameters).fit(X_trainval_scaled, Y_trainval)
    test_score = SelectedLogRegModel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedLogRegModel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    m = 'Logistic Regression (w/ imputation)'
    executionTime = "{:.2f}".format(time.perf_counter()-tps)
    acc.append([m, test_score, test_recall, test_auc, executionTime, fpr, tpr, thresholds])
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })


@app.route('/api/alzheimer/oasis/dataset/svm/rbf')
def rbf():
    best_score = 0
    k_parameter = 'rbf'

    for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter C
        for gamma_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter gamma
    #        for k_parameter in ['rbf', 'linear', 'poly', 'sigmoid']: # iterate over the values we need to try for the kernel parameter
                svmModel = SVC(kernel=k_parameter, C=c_paramter, gamma=gamma_paramter) #define the model
                # perform cross-validation
                scores = cross_val_score(svmModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
                # the training set will be split internally into training and cross validation

                # compute mean cross-validation accuracy
                score = np.mean(scores)
                # if we got a better score, store the score and parameters
                if score > best_score:
                    best_score = score #store the score 
                    best_parameter_c = c_paramter #store the parameter c
                    best_parameter_gamma = gamma_paramter #store the parameter gamma
                    best_parameter_k = k_parameter
    # rebuild a model with best parameters to get score 
    SelectedSVMmodel = SVC(C=best_parameter_c, gamma=best_parameter_gamma, kernel='rbf').fit(X_trainval_scaled_dna, Y_trainval_dna)

    test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })

@app.route('/api/alzheimer/oasis/dataset/svm/linear')
def linear():
    best_score = 0
    k_parameter='linear'
    degree_paramter = 1   ## degre du polynôme 

    for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter C
        for degree_paramter in [1, 2, 3, 4, 5]:
            svmModel = SVC(kernel=k_parameter, C=c_paramter, degree=degree_paramter) #define the model
            # perform cross-validation
            scores = cross_val_score(svmModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
            # the training set will be split internally into training and cross validation

            score = np.mean(scores)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score #store the score 
                best_parameter_c = c_paramter #store the parameter c
                best_parameter_degree = degree_paramter # store the parameter degree
    # rebuild a model with best parameters to get score 
    SelectedSVMmodel = SVC(C=best_parameter_c, degree=best_parameter_degree, kernel='linear').fit(X_trainval_scaled_dna, Y_trainval_dna)
    test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })

@app.route('/api/alzheimer/oasis/dataset/svm/poly')
def poly():
    best_score = 0
    k_parameter='poly'
    for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter C
        for gamma_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter gamma
            for degree_paramter in [1, 2, 3, 4, 5]:
                svmModel = SVC(kernel=k_parameter, C=c_paramter, gamma=gamma_paramter) #define the model
                # perform cross-validation
                scores = cross_val_score(svmModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
                # the training set will be split internally into training and cross validation

                # compute mean cross-validation accuracy
                score = np.mean(scores)
                # if we got a better score, store the score and parameters
                if score > best_score:
                    best_score = score #store the score 
                    best_parameter_c = c_paramter #store the parameter c
                    best_parameter_gamma = gamma_paramter #store the parameter gamma
                    best_parameter_degree = degree_paramter #store the parameter gamma
    # rebuild a model with best parameters to get score 
    SelectedSVMmodel = SVC(C=best_parameter_c, gamma=best_parameter_gamma, degree=best_parameter_degree, kernel='poly').fit(X_trainval_scaled_dna, Y_trainval_dna)
    test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })

@app.route('/api/alzheimer/oasis/dataset/svm/sigmoid')
def sigmoid():
    best_score = 0
    k_parameter='sigmoid'
    for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter C
        for gamma_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: #iterate over the values we need to try for the parameter gamma
            svmModel = SVC(kernel=k_parameter, C=c_paramter, gamma=gamma_paramter) #define the model
            # perform cross-validation
            scores = cross_val_score(svmModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
            # the training set will be split internally into training and cross validation

            # compute mean cross-validation accuracy
            score = np.mean(scores)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score #store the score 
                best_parameter_c = c_paramter #store the parameter c
                best_parameter_gamma = gamma_paramter #store the parameter gamma
    # rebuild a model with best parameters to get score 
    SelectedSVMmodel = SVC(C=best_parameter_c, gamma=best_parameter_gamma, kernel='sigmoid').fit(X_trainval_scaled_dna, Y_trainval_dna)

    test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })


@app.route('/api/alzheimer/oasis/dataset/knn')
def knn():
    best_score = 0
    for k_paramter in range(1, 5): #iterate over the values we need to try for the hyperparameter k
            neigh = neighbors.KNeighborsClassifier(n_neighbors=k_paramter)
            neigh.fit(X_trainval_scaled_dna, Y_trainval_dna)
            # perform cross-validation
            scores = cross_val_score(neigh, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
            # the training set will be split internally into training and cross validation

            # compute mean cross-validation accuracy
            score = np.mean(scores)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score #store the score 
                best_parameter_k = k_paramter #store the parameter k
    # rebuild a model with best parameters to get score 
    neigh = neighbors.KNeighborsClassifier(n_neighbors=best_parameter_k)
    neigh.fit(X_test_scaled, Y_test)
    test_score = neigh.score(X_test_scaled, Y_test)
    PredictedOutput = neigh.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })

@app.route('/api/alzheimer/oasis/dataset/decisionTree')
def decisionTree():
    tps = time.perf_counter()

    best_score = 0
    #best_parameter = 1

    for md in range(1, 9): # iterate different maximum depth values
        # train the model
        treeModel = DecisionTreeClassifier(random_state=0, max_depth=md, criterion='gini')
        # perform cross-validation
        scores = cross_val_score(treeModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
        
        # compute mean cross-validation accuracy
        score = np.mean(scores)
        
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameter = md
            
    # Rebuild a model on the combined training and validation set        
    SelectedDTModel = DecisionTreeClassifier(max_depth=best_parameter).fit(X_trainval_scaled_dna, Y_trainval_dna )

    test_score = SelectedDTModel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedDTModel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })


@app.route('/api/alzheimer/oasis/dataset/GaussianNB')
def GaussianNB():
    
    tps = time.perf_counter()

    # rebuild a model on the combined training and validation set
    SelectedModel = make_pipeline(StandardScaler(),
                                    GaussianNB()).fit(X_trainval_scaled_dna, Y_trainval_dna)

    test_score = SelectedModel.score(X_test_scaled_dna, Y_test_dna)
    PredictedOutput = SelectedModel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc })

@app.route('/api/alzheimer/oasis/dataset/Perceptron')
def Perceptron():

    tps = time.perf_counter()

    best_score=0
    kfolds=5 # set the number of folds

    for c in [0.001, 0.1, 1, 10, 100]:
        cvModel = make_pipeline(StandardScaler(), 
                                    Perceptron(alpha=c))
        # perform cross-validation
        scores = cross_val_score(cvModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
        
        # compute mean cross-validation accuracy
        score = np.mean(scores)
        
        # Find the best parameters and score
        if score > best_score:
            best_score = score
            best_parameters = c

    # rebuild a model on the combined training and validation set
    SelectedModel = make_pipeline(StandardScaler(), 
                                    Perceptron(alpha=best_parameters)).fit(X_trainval_scaled_dna, Y_trainval_dna)

    test_score = SelectedModel.score(X_test_scaled_dna, Y_test_dna)
    PredictedOutput = SelectedModel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    return jsonify({"Best accuracy on validation set is:": best_score ,"Best parameter for regularization (C) is: ": best_parameters,"Test accuracy with best C parameter is": test_score,"Test recall with the best C parameter is": test_recall,"Test AUC with the best C parameter is": test_auc,"Execution time:": executionTime })


@app.errorhandler(404)
def page_not_found(error):
    return 'This page does not exist', 404

if __name__ == "__main__":
    df = pd.read_csv('http://www.oasis-brains.org/pdf/oasis_longitudinal.csv')
    data = json.loads(df.to_json (orient='records'))
    app.run(debug=True)
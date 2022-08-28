# -*- coding: utf-8 -*-
"""
Spyder Editor

Terrorisit classification problem. Predict terrorist group name from features provided in global terroris database (GTD)

Dataset: https://www.kaggle.com/datasets/START-UMD/gtd?resource=download
"""

import pandas as pd
import numpy as np
import copy
import random

#Pre-processing Imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

#Sampling Imports
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn import pipeline

#Metric/Reporting Imports
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


#Model Imports
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

random.seed(1)

def main():
    """
    Build and compare several different ML models to find the best performing model for predicting terrorist group responsible for terroris attack
    Dataset is found here: https://www.kaggle.com/START-UMD/gtd
    Best performing models will then be inspected closer and hyper-parameter tuning carried out.

    Returns
    -------
    None.

    """
    
    data_set = pd.read_csv('globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1", low_memory=False)       
    t_dict = data_set['gname'].value_counts()          #Identify Most prolific terrorist groups   
    del t_dict['Unknown']   
    t_list = t_dict.keys()                      #List of all terrorist groups        
    num_T_groups = 200                          #Set number of target class values
    f_num = 10                                  #Set number of features to select
    m = DecisionTreeClassifier(class_weight='balanced')           #Model used in CV and Nested CV and Grid Search
    pg = {'max_depth':[50, 75, 100], 'max_features':[0.5, 0.6, 0.7, 0.8, 0.9]}  #Parameter grid for grid search
    rebalance_method = 'TOMEK'                  #Default Rebalancing Technique
    contamination = 0.0001                      #Isolation Forest Contamination parameter, used in pre processing      

    X, y, = pre_process(data_set, t_list, num_T_groups, f_num, cont=contamination)
 
    
 
    
    """
    ############################################################################################################
                                         Uncomment Tests To Run
    ############################################################################################################
    """
    
    'Uncomment Below to establish baseline'
    best_model = establish_baseline(X, y)
    
    'Uncomment Below to Test Rebalancing Techniques'
    # test_rebalance(X, y)

    'Uncomment Below to Compare Cost Sensative Learning Algorithms'
    # compare_cost_sensative(X, y)
    
    'Uncomment Below to Experiment on Top 3 performing models'
    # X,X_val, y, y_val = train_test_split(X, y, test_size=0.15)
    # X, y = rebalance_dataset(X, y, rebalance_method)
    # experimentation(X,X_val, y, y_val)
    
    'Uncomment Below to run Grid Search'
    # grid_search(X, y, m, pg, rebalance_method)
    
    'Uncomment Below to run Cross Fold Validation'
    # cf(X,y, m)
    
    'Uncomment Below to run Nested Cross Fold Vallidation'
    # ncf(X, y, m, pg, rebalance_method)                                     
        
    """
    ############################################################################################################
                            
    ############################################################################################################
    """

  
def compare_cost_sensative(X, y):
    """
    Compare Default Decision Tree Classifier and Cost Sensative Decision Tree Classifier.

    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.

    Returns
    -------
    f1 : FLOAT
        F1 Score for default Decision Tree Classifier.
    f1_cs : FLOAT
        F1 Score for cost sensative Decision Tree Classifier.
    """

    X,X_val, y, y_val = train_test_split(X, y, test_size=0.15)
    mod = DecisionTreeClassifier().fit(X, y)
    cs_mod = DecisionTreeClassifier(class_weight='balanced').fit(X, y)
    predictions = mod.predict(X_val)
    predictions_cs = cs_mod.predict(X_val)
    f1 = f1_score(y_val, predictions, average='micro')   
    f1_cs = f1_score(y_val, predictions_cs, average='micro')  
    print(f1, f1_cs)

    return f1, f1_cs

    

def test_rebalance(X, y):
    """
    Iterates through each rebalance method and runs it on the top 3 performing models by callnig experimentation function
    Notice a copy is made of the training data before rebalancing each time

    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.

    Returns
    -------
    None.

    """
    
    rebalance_methods = ['no rebalance', 'rus', 'ros', 'SMOTE', 'TOMEK', 'SMOTETomek',
                         'SMOTEENN', 'rous', 'ADASYN']
    X,X_val, y, y_val = train_test_split(X, y, test_size=0.15)
    for rebalance_method in rebalance_methods:
        print(rebalance_method)
        X1 = copy.deepcopy(X) 
        y1 = copy.deepcopy(y)
        X1, y1 = rebalance_dataset(X1, y1, rebalance_method)
        experimentation(X1,X_val, y1, y_val)
 


def rebalance_dataset(X,y, method='TOMEK'):
    """
    Perform rebalancing technique specified in the method parameter    
    
    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.
    method : STRING, optional
        'no rebalance', 'rus', 'ros', 'SMOTE', 'TOMEK', 'SMOTETomek',
        'SMOTEENN', 'rous', 'ADASYN'. The default is 'TOMEK'.

    Returns
    -------
    X : NUMPY ARRAY
        Rebalanced Training Data.
    y : NUMPY ARRAY
        Rebalanced Target Data.

    """
    
    'Dictionarys for random resampling - See accompanied report for more details (Methodology -> Research)'
    y = pd.Series(y)
    y_dict = y.value_counts()
    os_dict = {}
    us_dict = {}  
    os2_dict = {}
                                  #Creating 2 dictionarys for over sampling and under sampling strategys
    for k in list(y_dict.keys()):         #SMOTE by default requires 6 entries per class
        if y_dict[k] < 6:
            os_dict[k] = 6                           #Random Over Sampling strategy (6 instanes per class)
        if y_dict[k] > y_dict[0]/2:
            us_dict[k] = y_dict[0]/2             #Random Under Sampling strat 50% majority class
        if y_dict[k] < 100:                     #Random over sample to a minimum of 100 samples. Used in Random Over & Under sampler (rous)
            os2_dict[k] = 100
            
    if method == 'rus':
        X, y = RandomUnderSampler().fit_sample(X,y)
        # print(X.shape, y.shape)
    elif method == 'ros':
        X, y = RandomOverSampler().fit_sample(X,y)
        # print(X.shape, y.shape)
    elif method == 'SMOTE':
        if os_dict:                                                     #If minority class exists with less than 6 entries
            ros = RandomOverSampler(sampling_strategy=os_dict)          #Randomely over sample minority classes w/ <6 entries
            smt = SMOTE()
            pipe = pipeline.Pipeline([('os', ros), ('smote', smt)])     #Pipe ros into SMOTE
            X, y = pipe.fit_resample(X, y)
        else:
            X,y = SMOTE().fit_sample(X,y)        
        # print(X.shape, y.shape)
    elif method == 'TOMEK':                     
        X, y = TomekLinks().fit_sample(X,y)
        # print(X.shape, y.shape)
    elif method == 'SMOTETomek':       
        if os_dict:                                                     #if minority class exists with less than 6 entries
            ros = RandomOverSampler(sampling_strategy=os_dict)
            smt = SMOTETomek()
            pipe = pipeline.Pipeline([('os', ros), ('smote', smt)])
            X, y = pipe.fit_resample(X, y)
        else:
            X, y = SMOTETomek().fit_sample(X, y)
        # print(X.shape, y.shape)
    elif method == 'SMOTEENN':   
        if os_dict:    
            ros = RandomOverSampler(sampling_strategy=os_dict)
            smt = SMOTEENN()
            pipe = pipeline.Pipeline([('os', ros), ('smote', smt)])
        else:
            X, y = SMOTEENN().fit_sample(X, y)
        # print(X.shape, y.shape)    
    elif method == 'rous':
        #Random Over and Under Sampling
        ros = RandomOverSampler(sampling_strategy=os_dict)
        rus = RandomUnderSampler(sampling_strategy=us_dict)
        pipe = pipeline.Pipeline([('os', ros), ('us', rus)])
        X, y = pipe.fit_resample(X,y)
        # print(X.shape, y.shape)
    elif method == 'ADASYN':
        X, y = ADASYN().fit_sample(X,y)
        # print(X.shape, y.shape)
    else:
        pass
      
    return X, y 




def grid_search(X, y, model, param_grid, rebalance_method='TOMEK'):
    """
    Crossfold Validation strategy. 

    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.
    model : SKLearn Classifier Model
        
    rebalance_method : STRING, optional
        'no rebalance', 'rus', 'ros', 'SMOTE', 'TOMEK', 'SMOTETomek',
        'SMOTEENN', 'rous', 'ADASYN'. The default is 'TOMEK'.

    Returns
    -------
    None.

    """
    
    X,X_val, y, y_val = train_test_split(X, y, test_size=0.15)
    X, y = rebalance_dataset(X, y)
    cv = StratifiedKFold(n_splits=10, shuffle=True)                     #10 Splits for cross fold
    grid_search = GridSearchCV(model, param_grid, scoring='f1_micro', cv = cv, refit=True, n_jobs=-1)
    result = grid_search.fit(X, y)
    best_model=result.best_estimator_
    predictions = best_model.predict(X_val)
    f1 = f1_score(y_val, predictions, average='micro') 
    print("Best f1 Results: ", f1, "with parameters: ", result.best_params_)      
      
         
 

def cf(X, y, model, rebalance_method='TOMEK'):
    """
    Crossfold Validation strategy. 

    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.
    model : SKLearn Classifier Model
        
    rebalance_method : STRING, optional
        'no rebalance', 'rus', 'ros', 'SMOTE', 'TOMEK', 'SMOTETomek',
        'SMOTEENN', 'rous', 'ADASYN'. The default is 'TOMEK'.

    Returns
    -------
    None.

    """
     
    
    kf = StratifiedKFold(shuffle=True)
    results = []
    y = np.array(y)
    for train_i, test_i in kf.split(X, y):       
        train_X = copy.deepcopy(X[train_i])
        train_y = copy.deepcopy(y[train_i])
        train_X, train_y = rebalance_dataset(train_X, train_y)
        model.fit(train_X, train_y)    
        predictions = model.predict(X[test_i])
        f1 = f1_score(y[test_i], predictions, average='micro')
        results.append(f1)
    
    print(results)
    print("Mean: ", sum(results)/len(results))
    


def ncf(X, y, model, param_grid, rebalance_method='TOMEK'):                        
    """
    Nested cross-fold validation

    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.
    model : SKLearn Classifier Model
        Model to be used in grid search
    param_grid : DICT
        parameter : list of values for grid search CV
        eg. 'max_features':[0.5, 0.6, 0.7, 0.8, 0.9]
    rebalance_method : STRING, optional
        Rebalancing method to be used. The default is 'TOMEK'.

    Returns
    -------
    None.

    """
    
    cv_o = StratifiedKFold(n_splits=10, shuffle=True)
    results=[]    
    
    for train_i, test_i, in cv_o.split(X,y):     
        train_X = copy.deepcopy(X[train_i])
        train_y = copy.deepcopy(y[train_i])
        train_X, train_y = rebalance_dataset(train_X, train_y)
        cv_inner = StratifiedKFold(n_splits=10, shuffle=True)
        grid_search = GridSearchCV(model, param_grid, scoring='f1_micro', cv = cv_inner, refit=True, n_jobs=-1)
        result = grid_search.fit(train_X, train_y)
        best_model=result.best_estimator_
        predictions = best_model.predict(X[test_i])
        f1 = f1_score(y[test_i], predictions, average='micro') 
        results.append(f1)
        print("Best f1 Results: ", f1, "with parameters: ", result.best_params_)      
    print("Overall Accuracy: ", np.mean(results), np.std(results))             
     
      
   
def experimentation(X_train, X_test, y_train, y_test):
    """
    Runs Top 3 performing models on supplied data

    Parameters
    ----------
    X_train : NUMPY ARRAY
        Training Data.
    X_test : NUMPY ARRAY
        Validation data.
    y_train : NUMPY ARRAY
        Training Target data.
    y_test : NUMPY ARRAY
        Validation target data.

    Returns
    -------
    None.

    """
    
    models = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()] #Top performing models
    
    keys = []
    scores = {} 
    fit_models = {}
    best_model = None
    best_score = 0
    
    for model in models:
        m = model.fit(X_train, y_train)
        predictions = m.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')
        m_name = type(model).__name__
        scores[m_name] = f1
        fit_models[m_name] = m      
        keys.append(m_name)
    
    # print("\n\t\tML Model : F1 Score\n")
    for key in keys:
        # print("{:<25} : {}".format(key, scores[key]))
        print(scores[key])
        if scores[key] > best_score:
            best_score = scores[key]
            best_model = fit_models[key]     
    
    
    
    
  
def establish_baseline(X, y):
    """
    Runs models defined below in models list. Prints F1 score. Returns top performingg model

    Parameters
    ----------
    X : NUMPY ARRAY
        Training Data.
    y : NUMPY ARRAY
        Target Data.

    Returns
    -------
    best_model : SKLearn Classifier Object
        Best Performing Model.

    """
      
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    keys = []
    scores = {} 
    fit_models = {}
    best_model = None
    best_score = 0
    #Models to be tested
    models = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), SVC(), 
              SGDClassifier(), LogisticRegression(), GaussianNB(), AdaBoostClassifier()]
    
    #iterate through models, save f1 scores
    for model in models:
        m = model.fit(X_train, y_train)
        predictions = m.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')
        m_name = type(model).__name__
        scores[m_name] = f1
        fit_models[m_name] = m      
        keys.append(m_name)
        
    print("\n\t\tML Model : F1 Score")
    for key in keys:
        print("{:<25} : {}".format(key, scores[key]))
        if scores[key] > best_score:
            best_score = scores[key]
            best_model = fit_models[key]
    
    print("\nUsing default settings, {} had the highest F1 score of {}".format(type(best_model).__name__, best_score))    
    predictions = best_model.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    return best_model


    

def pre_process(data, t_list, k, feat_num, cont=0.0015):  
    """
    Pre process data for use in machine learning models

    Parameters
    ----------
    data : PANDAS DATAFRAME
        DATASET.
    t_list : LIST
        List of all Terrorist groups in dataset.
    k : INT
        Number of target values to retain.
    feat_num : INT
        Number of Features to be kept.
    cont : FLOAT, optional
        Contamination parameter for Isolation Forest. The default is 0.0015.

    Returns
    -------
    Train : NUMPY ARRAY
        Training Data.
    target : NUMPY ARRAY
        Target Data.

    """

    concat_ls=[]                            #Parse k number of groups from dataset
    for i in range(k):
        a = data.loc[data.gname==t_list[i]]
        concat_ls.append(a)
        data.gname[data.gname==t_list[i]]=i

    data = pd.concat(concat_ls)                         #Merge k terrorist groups data instances together
    Train = data.drop('gname', axis=1)                  
    target = data['gname']

    
    #Removing Categorical Features
    Train = Train.select_dtypes(include=['int64', 'float64', 'boolean'])        #Dataset has encoded all text values as numbers
    
    #Dealing Missing Values
    Train.replace('?',np.NaN,inplace=True)
    imp=SimpleImputer(missing_values=np.NaN)
    Train=pd.DataFrame(imp.fit_transform(Train))
    
    #Scaling
    scaler = StandardScaler()
    Train = scaler.fit_transform(Train)
    
    #Feature Selection
    ind = Train.shape[1] - feat_num                                 #Number of features to remove
    f =  RandomForestClassifier(n_estimators=250, random_state=0) 
    f.fit(Train,target) 
    importances = f.feature_importances_
    sortedIndices = np.argsort(importances)
    Train = np.delete(Train, sortedIndices[0:ind], axis=1)  

    #Dealing with Outliers
    ifor = IsolationForest(contamination=cont).fit(Train)      #Fit isolation forrest
    res = ifor.predict(Train)                                   #Find isoloated instances
    Train = Train[res==1]                                       #Parse non-isolated instances
    target = target[res==1]     
    sns.boxplot(data=Train)
    plt.show()
    
    target = np.array(target)
    
    return Train, target


if __name__=='__main__':
    main()
    
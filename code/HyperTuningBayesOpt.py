# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:38:54 2021

@author: andre.rizzo
"""


def Bayesian_Optimization_SVC(X_train, y_train):
    
    # Hyperparameter tuning - SVC
    from skopt import BayesSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.svm import SVC
    
    # define search space
    hparams = dict()
    hparams['C'] = (1e-6, 100.0, 'log-uniform')
    hparams['gamma'] = (1e-6, 100.0, 'log-uniform')
    hparams['degree'] = (1,5)
    hparams['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    
    # Define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Define Optimization Parameters
    search = BayesSearchCV(estimator=SVC(), search_spaces=hparams, n_jobs=-1, cv=cv)
    
    # Perform the search
    search.fit(X_train, y_train)
    
    # Store result
    SVC_hyper_result = search.best_params_
    
    # Report best result
    print("Best hyperparameters:")
    print(SVC_hyper_result, "\n")
    print("SVC model accuracy after best hyperparameter definition: %.3f" % (search.best_score_))
    


def Bayesian_Optimization_kNN(X_train, y_train):
    
    # Hyperparameter tuning - kNN
    from skopt import BayesSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier as clf
    
    # define search space
    hparams = dict()
    hparams['n_neighbors'] = (1, 20, 'uniform')
    hparams['weights'] = ('uniform', 'distance')
    hparams['p'] = (1, 2)
    
    # Define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    # Define the search
    search = BayesSearchCV(estimator=clf(), search_spaces=hparams, n_jobs=-1, cv=cv)
    
    # Perform the search
    search.fit(X_train, y_train)
    
    # Store result
    kNN_hyper_result = search.best_params_
    
    # Report best result
    print("Best hyperparameters:")
    print(kNN_hyper_result, "\n")
    print("kNN model accuracy after best hyperparameter definition: %.3f" % (search.best_score_))




def Bayesian_Optimization_RF(X_train, y_train):
    
    # Hyperparameter tuning - Random Forest Classifier
    from skopt import BayesSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import RandomForestClassifier as clf
    
    # define search space
    hparams = dict()
    hparams['n_estimators'] = (1, 1000, 'uniform')
    hparams['criterion'] = ['gini', 'entropy']
    hparams['max_leaf_nodes'] = (2,100, "uniform")
    
    # Define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Define the search
    search = BayesSearchCV(estimator=clf(), search_spaces=hparams, n_jobs=-1, cv=cv)
    
    # Perform the search
    search.fit(X_train, y_train)
    
    # Store result
    RF_hyper_result = search.best_params_
    
    # Report best result
    print("Best hyperparameters:")
    print(RF_hyper_result, "\n")
    print("Random Forest model accuracy after best hyperparameter definition: %.3f" % (search.best_score_))
        


def Bayesian_Optimization_LDA(X_train, y_train):
    
    # Hyperparameter tuning - LDA (using tune_sklearn)
    from skopt import BayesSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as clf
    
    # define search space
    hparams = dict()
    hparams['solver'] = ['lsqr', 'eigen']
    hparams['shrinkage'] = (None,'auto')
    #hparams['n_components'] = (1, 2)
    hparams['tol'] = (1e-6, 1e10, "log-uniform")
    
    # Define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Define the search
    search = BayesSearchCV(estimator=clf(), search_spaces=hparams, n_jobs=-1, cv=cv)
    
    # Perform the search
    search.fit(X_train, y_train)
    
    # Store result
    LDA_hyper_result = search.best_params_
    
    # Report best result
    print("Best hyperparameters:")
    print(RF_hyper_result, "\n")
    print("LDA model accuracy after best hyperparameter definition: %.3f" % (search.best_score_))
    
    
    
    
def Bayesian_Optimization_NaiveBayes(X_train, y_train):
    
    # Hyperparameter tuning - Naive Bayes
    from skopt import BayesSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.naive_bayes import GaussianNB as clf
    
    # define search space
    hparams = dict()
    #hparams['priors'] = 
    hparams['var_smoothing'] = (1e-6, 1e10, "log-uniform")
   
    
    # Define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Define the search
    search = BayesSearchCV(estimator=clf(), search_spaces=hparams, n_jobs=-1, cv=cv)
    
    # Perform the search
    search.fit(X_train, y_train)
    
    # Store result
    NaiveBayes_hyper_result = search.best_params_
    
    # Report best result
    print("Best hyperparameters:")
    print(RF_hyper_result, "\n")
    print("Naive Bayes model accuracy after best hyperparameter definition: %.3f" % (search.best_score_))


    
    
    
    
    
    
    
    
    
    
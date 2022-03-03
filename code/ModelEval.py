# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:40:46 2021

@author: andre.rizzo
"""

global results
results = {}

def LDA_Eval(X_train, y_train):
    
    # Evaluate Models - Linear Discriminant Analysis (LDA)
    
    # Import libraries
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = LDA()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Linear Discriminant Analysis model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['LDA'] = np.mean(scores)
    return results



def kNN_Eval(X_train, y_train):
    
    # Evaluate Models - k-Nearest Neighbors (kNN)

    # Import libraries
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = KNeighborsClassifier()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores= 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("k-Nearest Neighbors model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['kNN'] = np.mean(scores)
    return results


def SVC_Eval(X_train, y_train):
    
    # Evaluate Models - Support Vector Classifier (SVC)
    
    # Import libraries
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = SVC(kernel='linear')
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Support Vector Classifier model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['SVC'] = np.mean(scores)
    return results




def SGD_Eval(X_train, y_train):
    
    # Evaluate Models - Stochastic Gradient Descend (SGD)

    # Import libraries
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = SGDClassifier()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Stochastic Gradient Descend model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['SGD'] = np.mean(scores)
    return results




def NB_Eval(X_train, y_train):
    
    # Evaluate Models - Gaussian Naive Bayes (GNB)

    # Import libraries
    import numpy as np
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = GaussianNB()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Naive Bayes model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['Gaussian Naive Bayes'] = np.mean(scores)
    return results




def DTC_Eval(X_train, y_train):
    
    # Evaluate Models - Decision Tree Classifier (DTC)
    
    # Import libraries
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = DecisionTreeClassifier()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Decision Tree Classifier model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['Decision Trees'] = np.mean(scores)
    return results



def RF_Eval(X_train, y_train):
    
    # Evaluate Models - Random Forest
    
    # Import libraries
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = RandomForestClassifier()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1, error_score='raise')
    
    print("Random Forerst model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['Random Forest'] = np.mean(scores)
    return results



def BTC_Eval(X_train, y_train):
    
    # Evaluate Models - Bagging Tree Classifier (BTC)
    
    # Import libraries
    import numpy as np
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = BaggingClassifier()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1, error_score='raise')
    
    print("Bagging Tree Classifier model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['Bagging Tree'] = np.mean(scores)
    return results


def ADA_Eval(X_train, y_train):
    
    # Evaluate Models - AdaBoost

    # Import libraries
    import numpy as np
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = AdaBoostClassifier()
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1, error_score='raise')
    
    print("AdaBoost model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['AdaBoost'] = np.mean(scores)
    return results



def MLP_Eval(X_train, y_train):
    
    # Evaluate Models - Muli-Layer Perceptron Classifier (MLP)

    # Import libraries
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    classifier = MLPClassifier(max_iter=300)
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Multi-Layer Perceptron Model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['MLP Classifier'] = np.mean(scores)
    return results



def SkVC_Eval(X_train, y_train):
    
    # Evaluate Models - Sklearn's Voting Classifier (SkLC)

    # Import libraries
    import numpy as np
    from sklearn.ensemble import VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Define model
    clf1 = DecisionTreeClassifier()
    clf2 = GaussianNB()
    
    classifier = VotingClassifier(estimators=[('DTC', clf1), ('NB', clf2)], 
                       voting='soft', weights=[1,2])
    
    # Define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Evaluate model
    scores = 0
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("Sklearn's Voting Classifier Model accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)), "\n")
    
    # Update dictionary from accuracy results
    #results = {}
    results['SkVC Classifier'] = np.mean(scores)
    return results


def Print_Results(results):
    
    # Show results
    show_res = sorted(results.items(), reverse=True, key=lambda item: item[1])
      
    print(show_res)


def Save_Results(results, filename):
    
    # Show results
    import csv
    show_res = sorted(results.items(), reverse=True, key=lambda item: item[1])
    
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerows(show_res)
    
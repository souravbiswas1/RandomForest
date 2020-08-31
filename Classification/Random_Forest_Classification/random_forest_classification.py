# Random Forest Classification

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
Gender_encoded = pd.get_dummies(dataset['Gender'])
dataset = dataset.drop('Gender',axis = 1)
dataset = dataset.join(Gender_encoded)

X = dataset.iloc[:, [1, 2, 5]].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
# X_sc = sc.fit_transform(X)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 50, 
                                    criterion = 'gini', 
                                    random_state = 0,
                                    min_samples_leaf = 10,
                                    max_features = 'auto',
                                    )
classifier.fit(X_train_sc, y_train)

# Predicting the Test set results
y_test_pred = classifier.predict(X_test_sc)
# y_pred = classifier.predict(X_sc)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
def print_confusion_matrix(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    print('True positive = ', TP)
    print('True positive rate = ', TP * 100 / (TP + FN))
    print('False positive = ', FP)
    print('False positive rate = ', FP * 100 / (FP + TN))
    print('False negative = ', FN)
    print('False negative rate = ', FN * 100 / (FN + TP))
    print('True negative = ', TN)
    print('True negative rate = ', TN *100 / (TN + FP))
    print(cm)
    # print('Accuracy = ', (TP + TN) * 100 / (TP + FP + FN + TN))


#implementing K-fold cross validation---
def k_fold():
    accuracies = cross_val_score(estimator = classifier,
                                X = X_train,
                                y = y_train,
                                cv = 10,
                                n_jobs = -1)
    print('K-fold Accuracy : ',accuracies)
    print('K-fold Avg accuracy = ',accuracies.mean())
    print('K-fold Accuracy std deviation = ',accuracies.std())


#Applying Grid-search and find best models and parameters---
def grid_search():
    param_grid = { 
        'n_estimators': [10,50,100,200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [10,50,100]
    }
    CV_rfc = GridSearchCV(estimator = classifier, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y)
    print('Grid-search best parameters :')
    print (CV_rfc.best_params_)
    print('grid_search best score :')
    print(CV_rfc.best_score_)

print_confusion_matrix(y_test, y_test_pred)
k_fold()
grid_search()
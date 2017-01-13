# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 12:36:08 2016

@author: vijay Anand
"""

import pandas as pd
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier

###############################################################################

raw_train = pd.read_csv("train.csv")
raw_test = pd.read_csv("test.csv")

## remove unnecessary columns
train=raw_train.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)
test=raw_test.drop(['Name','Ticket','Cabin','Embarked'], axis=1)

## fill the empty fares with average fare values in test set

test["Fare"].fillna(test["Fare"].mean(), inplace=True)

## fill the empty age values with mean age

train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)


## Combine the parch and sinsp columns to a single column

train['Family']=0
test['Family']=0

train['Family'] =  train["Parch"] + train["SibSp"]
test['Family'] =  test["Parch"] + test["SibSp"]

train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0


train = train.drop(['SibSp','Parch'], axis=1)
test    = test.drop(['SibSp','Parch'], axis=1)

## Convert male to 0 and female to 1
train['Sex'].loc[train['Sex'] == 'male'] = 0
train['Sex'].loc[train['Sex'] == 'female'] = 1

test['Sex'].loc[test['Sex'] == 'male'] = 0
test['Sex'].loc[test['Sex'] == 'female'] = 1

## drop passenger ID 
PID=test['PassengerId']
print PID
test_final=test.drop(['PassengerId'], axis=1)

###############################################################################

train_labels= train["Survived"]
train_data = train.drop("Survived",axis=1)

### SVM
#svc = SVC(kernel='linear',gamma=5, C=50)
#svc.fit(train_data, train_labels)
#print 'Fit'
#Y_pred = svc.predict(test_final)
#print 'Pred'
#print svc.score(train_data, train_labels)

random_forest = RandomForestClassifier(n_estimators=1000,criterion='entropy')
random_forest.fit(train_data, train_labels)
Y_pred = random_forest.predict(test_final)
print random_forest.score(train_data, train_labels)

###############################################################################

submission = pd.DataFrame({
        "PassengerId": PID,
        "Survived": Y_pred
    })
    
submission.to_csv('titanic.csv', index=False)
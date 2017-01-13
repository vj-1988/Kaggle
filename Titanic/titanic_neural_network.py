# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:29:59 2016

@author: vijay

Simple neural network based model to predict survival in Titanic


"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
import time


###############################################################################

raw_train = pd.read_csv("train.csv")
raw_test = pd.read_csv("test.csv")

## remove unnecessary columns
train=raw_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test=raw_test.drop(['Name','Ticket','Cabin'], axis=1)

## fill the empty fares with average fare values in test set

test["Fare"].fillna(test["Fare"].mean(), inplace=True)
train["Fare"].fillna(train["Fare"].mean(), inplace=True)

## Embarked to integers

test["Embarked"].fillna('S')
train["Embarked"].fillna('S')

test['Embarked'].loc[test['Embarked'] == 'S'] = 0
test['Embarked'].loc[test['Embarked'] == 'C'] = 1
test['Embarked'].loc[test['Embarked'] == 'Q'] = 2

train['Embarked'].loc[train['Embarked'] == 'S'] = 0
train['Embarked'].loc[train['Embarked'] == 'C'] = 1
train['Embarked'].loc[train['Embarked'] == 'Q'] = 2

## fill the empty age values with mean age

train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)
train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)

## Combine the parch and sinsp columns to a single column

train['Family']=0
test['Family']=0
train['Adult']=0
test['Adult']=0


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

test['Adult'].loc[train['Age'] > 16] = 1
test['Adult'].loc[test['Age'] > 16] = 1

## drop passenger ID 
PID=test['PassengerId']
test_final=test.drop(['PassengerId'], axis=1)

###############################################################################

train_labels= train["Survived"]
train_data = train.drop("Survived",axis=1).iloc[:,:].values

###############################################################################

model = Sequential()
model.add(Dense(8, input_dim=7, init='uniform', activation='relu'))
model.add(Dense(256, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
# Fit the model
model.fit(train_data, train_labels, nb_epoch=4000, batch_size=4000)

scores = model.evaluate(train_data, train_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



###############################################################################
#### prediction

test_final_np=test_final.iloc[:,:].values
predictions = model.predict(test_final_np)
rounded = [int(round(x)) for x in predictions]
print(rounded)

###############################################################################

submission = pd.DataFrame({
        "PassengerId": PID,
        "Survived": rounded
    })
    
submission.to_csv('titanic.csv', index=False)

print 'Done!'
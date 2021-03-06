## import section as basic inputs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##data collection
Data = pd.read_csv('monodata.csv', header=None) ## not to remove the first row
# Data = pd.read_csv('bidata.csv', header=None) ## not to remove the first row
# Data = pd.read_csv('tridata.csv', header=None) ## not to remove the first row

Data = Data.drop_duplicates() ## to get the uniqe value and romove duplicate values that leads to a overfitting problem

row,col = Data.shape
print(Data.shape) ##optional for geting to know the (row,col) of the datasets
#
#  print(Data) ## to see the data set
#
# ##check for any missing value in the dataset
# X = Data.iloc[:, 0:col-1]
# y = Data.iloc[:, col-1]
# print("there is null value in the feature list=> ",any(X.isnull()))
# print("there is null vaue  in the label=> ",any(y.isnull()))

#
# ##divide the feature X and the label y col

X = Data.iloc[:, 0:col-1].values
y = Data.iloc[:, col-1].values

#
# ## find the missing values and fit them as per mean,median,mode
from sklearn.preprocessing import Imputer
# X[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]=Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True).fit_transform\
#     (X[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])
##loop Method
X[:,:]  = Imputer(missing_values="NaN", strategy="mean", axis=0).fit_transform(X[:,:])


##dataset spliting
from sklearn.model_selection import cross_val_predict


#featture scalling
from sklearn.preprocessing import StandardScaler, Normalizer
## StandardScaler the values are in between [-1,1]
## Normalizer the value of each feature will be in between [0,1]

scale = StandardScaler()
X_scale = scale.fit_transform(X)


## Its time to use the Algorithms :D
##calling the algorithms from library
from sklearn.neighbors import KNeighborsClassifier

##model learning as per the algorithms
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski', p=2)
# model = model.fit(X_train,y_train)
cross_validation = cross_val_predict(model,X_scale,y,cv=10)

##making a new label:: y_predicted from X_test from the learned model.

##now to evaluate the acuurecy and others
from sklearn.metrics import accuracy_score
nameOfClassifer = model.__class__.__name__

print(nameOfClassifer)
print('accuracy= {}'.format(cross_validation.mean()*100))

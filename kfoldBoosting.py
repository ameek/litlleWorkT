# AdaBoost Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier




import pandas as pd
import matplotlib.pyplot as plt

##data collection
Data = pd.read_csv('monodata.csv', header=None) ## not to remove the first row
#Data = pd.read_csv('bidata.csv', header=None) ## not to remove the first row
#Data = pd.read_csv('tridata.csv', header=None) ## not to remove the first row

#Data = Data.drop_duplicates() ## to get the uniqe value and romove duplicate values that leads to a overfitting problem

row,col = Data.shape

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

#X[:,:]  = Imputer(missing_values="NaN", strategy="mean", axis=0).fit_transform(X[:,:])






seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
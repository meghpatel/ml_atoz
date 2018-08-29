import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing the dataset

dataset = pd.read_csv('Data.csv')


#seperating Dependent and Independent Variables
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


#Missing values in a dataset
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Laconic Code for the above 2 lines
X[:,1:3] = imputer.fit_transform(X[:,1:3])


#Dealing with Categorical Data (Method 1:Biased with values)
lbenc = LabelEncoder()
X[:,0] = lbenc.fit_transform(X[:,0])
Y[:] = lbenc.fit_transform(Y[:])

#OneHotEncoder(Not biased)
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()


#Splitting the Training and Test Dataset
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2,random_state=0)


#Feature Scaling
sc_X=StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain)
Xtest = sc_X.transform(Xtest)
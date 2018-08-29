import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

# Random forest classifier on titanic data set

# Import the libraries
import numpy as np
import matplotlib as plot
import pandas as pd
import math

# Import the data
dataset = pd.read_csv('titanic.csv')
indexes = [2, 3, 6]
X = dataset.iloc[:, indexes].values
Y = dataset.iloc[:, 5].values

# Take care of missing ages
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1].reshape(-1, 1))
X[:, 1] = imputer.transform(X[:, 1].reshape(-1, 1)).flatten()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit Multiple Linear Regression to data
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
regressor.fit(X_train, Y_train)

# Predict test results
right = 0
wrong = 0
y_pred = regressor.predict(X_test)
for i in range(0, len(y_pred)):
	if y_pred[i] == Y_test[i]:
		right += 1
	else:
		wrong += 1

total = len(y_pred)
print "Reliability: " + str(float(right)/float(total) * 100) + "%"

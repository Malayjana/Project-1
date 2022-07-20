import pandas as pd
import numpy as np
from pickle import dump
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor

#Reading the file from the local machine
data = pd.read_csv('D:\Project\energy_production.csv', delimiter = ';')

#Defining X & Y values
x = data.drop(['energy_production'], axis=1)
y = data['energy_production']

#Normalizing the data
from sklearn import preprocessing
data_norm = preprocessing.normalize(x)
names = ['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']
x_norm = pd.DataFrame(data_norm, columns=names)

# performing the train test split on the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_norm,y,test_size=0.20, random_state=24)

#Building the model
xgb = XGBRegressor(learning_rate=0.07,max_depth=7,objective='reg:linear', n_estimators=300)
xgb.fit(x_train , y_train)

#Saving the model
dump(xgb, open('xgb_save.sav','wb'))


# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NAVEEN KUMAR
RegisterNumber:  212222230091


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
*/
```

## Output:
data.head():

![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/fb0ce862-a96b-4fd9-bb40-e4ba07c1d6b3)

data.info():

![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/3b265222-069e-4f88-865e-4fa29e0e2643)

isnull() & sum() function:

![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/30787b4e-251e-4cf7-8560-8aee9f94138a)

data.head() for position:

![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/05d75f8b-9040-457e-a2df-3d481beee5c0)

MSE value:

![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/71c9c353-3a72-438b-aa12-baf2502161ba)

R2 value:

![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/03fcbdbc-b146-47f3-8e41-25b58f11324b)

Prediction value:
![image](https://github.com/mrnaviz/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/123350791/d85d6960-8a60-4645-945e-0e4e993ec79e)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

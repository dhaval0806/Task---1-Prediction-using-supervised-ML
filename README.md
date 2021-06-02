# Task---1-Prediction-using-supervised-ML
This is the task 1 given by the TSF as the part of my internship programme.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
print(dataset)

dataset.head()

dataset.plot(x='Hours',y='Scores',style='o')
plt.title('2-D visulization of dataset')
plt.show()

A=dataset.iloc[:, :-1].values
B=dataset.iloc[:, -1].values

A_train,A_test,B_train,B_test=train_test_split(A,B,test_size=0.2,random_state=0)
regressor_var=LinearRegression()
regressor_var.fit(A_train.reshape(-1,1),B_train)

line=A*regressor_var.coef_+regressor_var.intercept_
plt.scatter(A,B)
plt.plot(A,line,color='red')
plt.show()print(A_test)
B_pred=regressor_var.predict(A_test)

dataset2= pd.DataFrame({'Actual set':B_test, 'predicted':B_pred})
print(dataset2)

print('Training score=',regressor_var.score(A_train,B_train))
print('Predicting score=',regressor_var.score(A_test,B_test))

dataset2.plot(kind='line',figsize=(8,6))
plt.grid(which='major',linewidth='0.5',color='orange')
plt.grid(which='major',linewidth='0.5',color="blue")
plt.show()

hours=9.25
new=np.array([hours])
new=new.reshape(-1,1)
own_pred=regressor_var.predict(new)
print("No. of Hours :", hours)
print("Predicted Score :",own_pred)

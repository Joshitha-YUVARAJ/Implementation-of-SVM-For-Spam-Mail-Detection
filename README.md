# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2. Import the dataset to operate on.

3. Split the dataset.

4. Predict the required output.

5. End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: YUVARAJ JOSHITHA
RegisterNumber: 212223240189  
*/
import pandas as pd
data = pd.read_csv("/content/spam.csv",encoding = 'Windows-1252')
from sklearn.model_selection import train_test_split
data
data.info()
data.isnull().sum()
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

 DATASET:
 
 ![image](https://github.com/Joshitha-YUVARAJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742770/5dbea7c3-18e1-4387-81f5-33613258f3c9)
 
 DATA INFO:
 
 ![image](https://github.com/Joshitha-YUVARAJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742770/da0eb139-fd8f-40d0-8374-5377a5316db4)

Y_PREDICT:

![image](https://github.com/Joshitha-YUVARAJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742770/d90dfd46-ac69-4aa9-9bce-2d7567050b33)

ACCURACY:

![image](https://github.com/Joshitha-YUVARAJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742770/f3659df2-1934-401a-8411-147a0e63a1ea)

CONFUSION MATRIX:

![image](https://github.com/Joshitha-YUVARAJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742770/7b402c17-885c-4c85-8937-bdd7ff989114)

CLASSIFICATION REPORT:

![image](https://github.com/Joshitha-YUVARAJ/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742770/e5be84ae-8492-4020-967c-9834f0136746)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
filePath= '/Users/HP/Desktop/AI ML/heart.csv.zip'
data = pd.read_csv(filePath)
p=data.head(5)
print(p);
print("(Rows, columns): " + str(data.shape))
c=data.columns
print(c);
# returns the number of unique values for each variable.
a=data.nunique(axis=0)
print(a);
#summarizes the count, mean, standard deviation, min, and max for numeric variables.
d=data.describe()
print(d);
# it will show that if any place is empty 
print(data.isna().sum()) 
#modeling
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) # get instance of model
model5.fit(x_train, y_train) # Train/Fit model 

y_pred5 = model5.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred5)) # output accuracy   
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred5)
print(cm)
accuracy_score(y_test, y_pred5)   
print(accuracy_score);

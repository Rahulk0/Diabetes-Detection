import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"/home/friday/Downloads/diabetes.csv")
scaler=StandardScaler()
scaler.fit(data)
data.head()

data.isnull().sum()

sns.countplot(x='Outcome',data=data)
columns=data.columns[:8]
length=len(columns)
print(columns)
print(length)

sns.heatmap(data.corr(),annot=True)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

y=data["Outcome"]
x=data.iloc[:,:8]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train,y_train)

#random forest 
rf = RandomForestClassifier(n_estimators=300,random_state=72)
model_rf = rf.fit(x_train,y_train)
pred_rf = model_rf.predict(x_test)
print(confusion_matrix(y_test,pred_rf))
print(accuracy_score(pred_rf,y_test))
print(f1_score(pred_rf,y_test))
print(classification_report(pred_rf,y_test))

#logistic regression
model = LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(accuracy_score(prediction,y_test))
print(f1_score(prediction,y_test))
print(classification_report(prediction,y_test))

# KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5,p=2)
model.fit(x_train,y_train)
prediction1=model.predict(x_test)
print(confusion_matrix(y_test,prediction1))
print(accuracy_score(prediction1,y_test))
print(f1_score(prediction1,y_test))
print(classification_report(prediction1,y_test))

li=[]
leng=len(y_test)
for i in range(leng):
    li.append(i)
plt.figure(figsize=(15,8))
plt.scatter(li[:50],y_test[:50],marker='*')
plt.plot(li[:50],pred_rf[:50],'-',c='g',linewidth=6)
plt.plot(li[:50],prediction[:50],'-.',c='b',linewidth=4)
plt.plot(li[:50],prediction1[:50],':',c='r',linewidth=2)

plt.show()


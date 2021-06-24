#MALWARE DETECTION USING MACHINE LEARNING OTALI#
#COMPARING ALGORITHMS THAT ARE BASED FOR THIS TYPES OF PREDICTIONS#
#sklearn has lots of tools to anaylze data analysis. Here, we used one of them, train_test_split.#
from sklearn.model_selection import train_test_split
#Here, accuracy_score to find accuracy#
from sklearn.metrics import accuracy_score
#pandas to read dataset#
import pandas as pd
#numpy to convert data#
import numpy as np
#Here, I read data from our dataset comma separated file and print it.#
from matplotlib import pyplot as plt
a=pd.read_csv("test_permissions_new.csv")
print(a)
#Here, we take only the FEATURES and print x. 'type' is a label(ML Algroithm).# 
# The label taken from data set will be stored in Y. Axis=1 represents that I need columns.#
X=a.drop(['type'],axis=1)
print(X)
############################### labels ###############################
Y=a['type']
print(Y)
############################### training and testing part ###############################
#Here, I use train_test_split to split the dataset into training and testing.#
#If user select shuffle as true, it will shuffle. If user select it as false, it will order in ascending mode.#
x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=True,test_size=0.25, random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
############################### Decision Tree ###############################
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree.fit(x_train,y_train)#x_train is features, y_train is labels. Fit to train the machine#
p=decision_tree.predict(x_test)#predict for give an input and get an output for that input#
print("\n Decision Tree: \n ",p)
b=accuracy_score(p, y_test)
print("\n Decision Tree=",b)
############################### KNN ###############################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
pr = knn.predict(x_test)
print("\n KNN:\n",pr)
c=accuracy_score(pr, y_test)
print("\n KNN=",c)
############################### SVM ###############################
from sklearn.svm import LinearSVC
clf_lsvc = LinearSVC(random_state=0)
clf_lsvc.fit(x_train,y_train)
pred=clf_lsvc.predict(x_test)
print("\n Support Vector Machine: \n",pred)
d=accuracy_score(pred, y_test)
print("\n Support Vector Machine=",d)
############################### GN ###############################
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
predicted=model.predict(x_test)
print("\n Naive Bayes: \n",predicted)
e=accuracy_score(predicted, y_test)
print("\n Naive Bayes = ",e)
############################### Random Forest ###############################
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("\n Random Forest: \n",y_pred)
rf_acc=accuracy_score(y_pred, y_test)
print("\n Random Forest = ",rf_acc)

li=['Decision Tree','KNN','SVM','NB','RBF']
acc=[b,c,d,e,rf_acc]#b for DT, c for KNN...#
plt.bar(li,acc,width=.3)#Here, I used matplotlib to create graphics#
plt.show()
plt.bar()

##THIS PART IS FOR PREDICTION, NO NEED.#
##rr=pd.read_csv('Book1.csv')
##type(rr)
##y_pre=clf_lsvc.predict(rr)
##y_pred=str(y_pre)
##y_pred=str(y_pre)
##print (y_pred)

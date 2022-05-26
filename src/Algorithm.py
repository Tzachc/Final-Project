import dataset as dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

read_file = pd.read_excel (r'CVRI_data.xlsx')
read_file.to_csv (r'CVRI_model.csv', index = None, header=True)

dataset = pd.read_csv('CVRI_model.csv')
dataset.head()

x = dataset[['-20','-19','-18','-17','-16','-15','-14','-13','-12','-11','-10','-9','-8']]
y = dataset['event']

#Spliting data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=65)


#LogisticRegression
model_Log= LogisticRegression(random_state=1)
model_Log.fit(X_train,Y_train)
Y_pred= model_Log.predict(X_test)
model_Log_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy
print(model_Log_accuracy)

# Desicion tree
model_tree=DecisionTreeClassifier(random_state=10,criterion="gini",max_depth=100)
model_tree.fit(X_train,Y_train)
Y_pred=model_tree.predict(X_test)
model_tree_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy
print(model_tree_accuracy)

# Random forest
model_random= RandomForestClassifier(n_estimators= 10, criterion="gini")
model_random.fit(X_train,Y_train)
y_pred= model_random.predict(X_test)
model_random_accuracy=round(accuracy_score(Y_test,y_pred), 4)*100 # Accuracy
print(model_random_accuracy)

# xgboost
model_boost = XGBClassifier()
model_boost.fit(X_train, Y_train)
y_pred = model_boost.predict(X_test)
model_boost_accuracy=round(accuracy_score(Y_test,y_pred), 4)*100 # Accuracy
print(model_boost_accuracy)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
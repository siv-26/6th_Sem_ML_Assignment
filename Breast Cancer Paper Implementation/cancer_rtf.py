import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Import Cancer data from the Sklearn library

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()

df_cancer.shape

df_cancer.columns

sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'] )

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )

df_cancer['target'].value_counts()

sns.countplot(df_cancer['target'], label = "Count")

plt.figure(figsize=(20,12))
sns.heatmap(df_cancer.corr(), annot=True)

X = df_cancer.drop(['target'], axis = 1)
X.head()

y = df_cancer['target']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
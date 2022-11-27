import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
from sklearn.preprocessing import LabelEncoder
#import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

import pickle
df=pd.read_csv("diabetes.csv")
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_columns].values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Instantiate learning model (k = 4)
classifier = KNeighborsClassifier(n_neighbors=6)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracy= accuracy_score(y_test, y_pred)*100
print('Accuracy of your model is equal ' + str(round(accuracy, 2)) + ' %.')



pickle.dump(classifier, open("model.pkl","wb"))
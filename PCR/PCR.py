import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import sklearn.model_selection as skm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from yellowbrick.model_selection import learning_curve


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle as shf
import pickle
import os
import glob as gb

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score




code = {'NORMAL':0 ,'PNEUMONIA':1}
#function to return the class of the images from its number, so the function would return 'Normal' if given 0, and 'PNEUMONIA' if given 1.
def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x
        
#X_train, X_test contain the images as numpy arrays, while y_train, y_test contain the class of each image 
loaded_X_train = np.load('/Users/conny/Desktop/QTM 347/FinalProject/X_train.npy')
loaded_X_test = np.load('/Users/conny/Desktop/QTM 347/FinalProject/X_test.npy')
loaded_y_train = np.load('/Users/conny/Desktop/QTM 347/FinalProject/y_train.npy')
loaded_y_test = np.load('/Users/conny/Desktop/QTM 347/FinalProject/y_test.npy')
loaded_X_test = loaded_X_test[...,0]
loaded_X_train = loaded_X_train[...,0]

X_test = loaded_X_test.reshape(624, 64*64)
X_train = loaded_X_train.reshape(5216, 64*64)

y_train = loaded_y_train
y_test = loaded_y_test

#Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


warnings.filterwarnings("ignore", category=ConvergenceWarning)

num_components_PCA = np.arange(10, 200, 10)
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=123)

best_num_components = None
best_val_accuracy = 0
pcr_val_accuracy = []

for n in num_components_PCA:
    # Create a pipeline with PCA and logistic regression
    pca = PCA(n_components=n)
    logistic_reg = LogisticRegression()
    pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])

    scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring='accuracy')

    mean_accuracy = np.mean(scores)
    pcr_val_accuracy.append(mean_accuracy)

    if mean_accuracy > best_val_accuracy:
        best_val_accuracy = mean_accuracy
        best_num_components = n

# After the loop, best_num_components will have the number of components with the highest mean accuracy
print(f"Best Number of Components: {best_num_components}, with an average accuracy of: {best_val_accuracy}")


pcr_val_error = 1 - np.array(pcr_val_accuracy)
bst_pcr_val_error = 1 - best_val_accuracy
print(f"Best Validation Error: {bst_pcr_val_error}")


# Plot the validation error as a function of the number of components
plt.figure(figsize=(10,6))
plt.plot(np.arange(10, 200, 10), pcr_val_error, marker='o', linestyle='--', color='r')
plt.xlabel('Number of Components')
plt.ylabel('Validation Error')
plt.title('Validation Error vs Number of Components')
plt.show()



#Use the best number of components to fit the PCA model
pca = PCA(n_components=best_num_components)
pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
print(f"Test Error of PCR using the best number of components(140): {1 - pipe.score(X_test, y_test)}")




#function to plot the confusion matrix for each model
def plot_cm(predictions, y_test, title):
  labels = ['Normal', 'Pnuemonia']
  labels_predicted = ['Predicted Normal', 'Predicted Pnuemonia']
  labels_actual = ['Actual Normal', 'Actual Pnuemonia']
  cm = confusion_matrix(y_test,predictions)
  cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
  plt.figure(figsize = (8,8))
  plt.title(title)
  sns.heatmap(cm, linecolor = 'black' , linewidth = 2 , annot = True, fmt='', xticklabels = labels_predicted, yticklabels = labels_actual)
  plt.show()


pipe_pred_pcr = pipe.predict(X_test)
#plot confusion matrix for each model
plot_cm(pipe_pred_pcr, y_test, 'PCR Confusion Matrix')




smote = SMOTE(random_state=123)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(X_train_smote.shape)
print(y_train_smote.shape)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

num_components_PCA = np.arange(10, 200, 10)
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=123)

best_num_components = None
best_val_accuracy = 0
pcr_val_accuracy = []

for n in num_components_PCA:
    # Create a pipeline with PCA and logistic regression
    pca = PCA(n_components=n)
    logistic_reg = LogisticRegression()
    pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])

    scores = cross_val_score(pipe, X_train_smote, y_train_smote, cv=kfold, scoring='accuracy')

    mean_accuracy = np.mean(scores)
    pcr_val_accuracy.append(mean_accuracy)

    if mean_accuracy > best_val_accuracy:
        best_val_accuracy = mean_accuracy
        best_num_components = n

# After the loop, best_num_components will have the number of components with the highest mean accuracy
print(f"Best Number of Components: {best_num_components}, with an average test error of: {1- best_val_accuracy}")

#Use the best number of components to fit the PCA model
pca = PCA(n_components=best_num_components)
pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])
pipe.fit(X_train_smote, y_train_smote)
pipe.score(X_test, y_test)
print(f"Test Error of PCR using the best number of components(190): {1 - pipe.score(X_test, y_test)}")

pipe_pred_pcr_smote = pipe.predict(X_test)
plot_cm(pipe_pred_pcr_smote, y_test, 'PCR Confusion Matrix')







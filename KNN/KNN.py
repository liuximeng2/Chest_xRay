import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import sklearn.model_selection as skm

y_train_path = '/Users/hk_cheng/Downloads/y_train.npy'
x_train_path = '/Users/hk_cheng/Downloads/x_train.npy'
y_test_path = '/Users/hk_cheng/Downloads/y_test.npy'
x_test_path = '/Users/hk_cheng/Downloads/x_test.npy'

# Loading the arrays
y_train = np.load(y_train_path)
x_train = np.load(x_train_path)
y_test = np.load(y_test_path)
x_test = np.load(x_test_path)

Kfold = skm.KFold(n_splits = 5, shuffle = True, random_state = 123)

X_train = x_train.reshape([-1, np.product((64, 64, 3))])
X_test = x_test.reshape([-1, np.product((64, 64, 3))])

param_grid = {'n_neighbors': list(range(1, 101))}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=3)

grid_search.fit(X_train, y_train)

results = grid_search.cv_results_
n_neighbors = list(range(1, 101))
accuracies = results['mean_test_score']

plt.figure(figsize=(12, 6))
plt.plot(n_neighbors, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy for Different n_neighbors in KNN')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

best_n_neighbors = grid_search.best_params_['n_neighbors']
print(f"Best n_neighbors: {best_n_neighbors}")

knn = KNeighborsClassifier(n_neighbors= 4).fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)

print(f"Accuracy: {accuracy*100:.2f}%")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

csv_path = r"C:\Users\aasri\Desktop\ASSIGNMENT_5\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

df['X'] = df['Ingredients'].apply(len)
df['Y'] = df['Instructions'].astype(str).apply(len)  

threshold = 8 
df['Class'] = np.where(df['X'] + df['Y'] > threshold, 1, 0)

X = df[['X', 'Y']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier()

param_grid = {'n_neighbors': list(range(1, 21))}  

grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']

print(f'Best k value: {best_k}')

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)
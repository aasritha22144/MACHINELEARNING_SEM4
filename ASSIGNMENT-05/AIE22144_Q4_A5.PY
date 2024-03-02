import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

csv_path =r"C:\Users\aasri\Desktop\ASSIGNMENT_5\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

df['X'] = df['Ingredients'].apply(len)
df['Y'] = df['Instructions'].astype(str).apply(len)  

threshold = 8  
df['Class'] = np.where(df['X'] + df['Y'] > threshold, 1, 0)

X_test = np.arange(0, 10.1, 0.1)
Y_test = np.arange(0, 10.1, 0.1)
test_grid = np.array(np.meshgrid(X_test, Y_test)).T.reshape(-1, 2)

test_data = pd.DataFrame(test_grid, columns=['X', 'Y'])

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(df[['X', 'Y']], df['Class'])
test_data['Predicted_Class'] = knn_classifier.predict(test_data[['X', 'Y']])

colors = {0: 'blue', 1: 'red'}
plt.scatter(test_data['X'], test_data['Y'], c=test_data['Predicted_Class'].map(colors),
            label=test_data['Predicted_Class'].map({0: 'Class 0 (Blue)', 1: 'Class 1 (Red)'}))
plt.scatter(df['X'], df['Y'], c=df['Class'].map(colors), marker='x', label=df['Class'].map({0: 'Class 0 (Blue)', 1: 'Class 1 (Red)'}))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Test Data')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_excel(r"C:\Users\aasri\Desktop\ASSIGNMENT _ 6\Transactions.xlsx")

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def perceptron_classification(X, y, weights, learning_rate, max_epochs):
    for epoch in range(max_epochs):
        for x, target in zip(X.values, y):
            output = sigmoid_activation(np.dot(weights, x))
            error = target - output
            weights += learning_rate * error * x
    return weights

initial_weights = np.array([0.5, 0.3, 0.2])

learning_rate = 0.01

max_epochs = 1000

trained_weights = perceptron_classification(X_train, y_train, initial_weights, learning_rate, max_epochs)

predictions = [sigmoid_activation(np.dot(trained_weights, x)) for x in X_test.values]

accuracy = np.mean((predictions >= 0.5) == y_test.values)
print("Accuracy:", accuracy)

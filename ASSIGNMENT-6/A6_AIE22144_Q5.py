import numpy as np

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def perceptron_customer_classification(X, y, weights, learning_rate, max_epochs):
    errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for x, target in zip(X, y):
            output = sigmoid_activation(np.dot(weights, x))
            error = target - output
            total_error += error**2
            weights += learning_rate * error * x
        errors.append(total_error)
        if total_error <= 0.002:
            break
    return weights, errors

X_customer = np.array([[20, 6, 2], [16, 3, 6], [27, 6, 2], [19, 1, 2], [24, 4, 2], [22, 1, 5], [15, 4, 2], [18, 4, 2], [21, 1, 4], [16, 2, 4]])
y_customer = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

weights_initial = np.random.rand(3)

learning_rate = 0.1
max_epochs = 100

final_weights, errors = perceptron_customer_classification(X_customer, y_customer, weights_initial, learning_rate, max_epochs)

print("Final Weights:", final_weights)
print("Errors during training:", errors)

import numpy as np
import matplotlib.pyplot as plt

#generate data
np.random.seed(42)
n_samples = 100
n_features = 3

X = 2 * np.random.rand(n_samples, n_features)
true_weights = np.array([2,-3.5,-1.5])
bias =5
noise= np.random.rand(n_samples)

#true model
y = X @ true_weights + bias + noise

#split
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

#params
weights = np.random.random(n_features)
bias = np.random.random()

learning_rate = 0.01
epochs = 1000

#gradient descent
def gradient_descent(X, y, weights, bias, learning_rate):
    n = len(X)
    y_pred = X @ weights + bias

    #gradient
    dw = -(2/n) * X.T @ (y - y_pred)
    db = -(2/n) * np.sum(y - y_pred)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    return weights, bias

#training loop
for epoch in range(epochs):
    weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate)
    mse_train = np.mean(y_train - (X_train @ weights + bias) **2)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training MSE: {mse_train:.4f}")

#test
y_pred_test = X_test @ weights + bias
mse_test = np.mean((y_test - y_pred_test) ** 2)

print("\nFinal Parameters:")
print(f"Weights: {weights}")
print(f"Bias: {bias:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Visualize actual vs. predicted (test set)
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
plt.scatter(range(len(y_test)), y_pred_test, color="red", label="Predicted")
plt.xlabel("Sample Index")
plt.ylabel("y")
plt.legend()
plt.title("Actual vs Predicted (Test Set)")
plt.show()

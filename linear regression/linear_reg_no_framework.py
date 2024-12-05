import numpy as np
import matplotlib.pyplot as plt

#generate synthetic datta
np.random.seed(42)  # initiate the random
X = 2 * np.random.rand(100, 1) #100 data points, single feature
y = 4 + 3 * X + np.random.randn(100, 1) #model y=4 + 3x + noise

X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)


#train and test split
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test  = X[:split_index] ,  X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


#params
w = np.random.rand(1) #random weight
b = np.random.rand(1) #random bias
learning_rate = 0.001
epochs = 1000

#loss function MSE - gradient descent
def gradient_descent(X, y, w, b, learning_rate):
    n = len(X)
    y_pred = X*w + b

    #partial derivatives - calculating gradient - negative so it can move in the descent direction
    dw = -(2 / n) * np.sum(X * (y - y_pred))  # Partial derivative w.r.t. w
    db = -(2 / n) * np.sum(y - y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b

training_losses = []
#training
for epoch in range(epochs):
    w,b = gradient_descent(X_train, y_train, w, b, learning_rate)
    mse_train = np.mean((y_train - (w * X_train + b)) ** 2)
    training_losses.append(mse_train)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, w: {w[0]:.4f}, b: {b[0]:.4f}, Training MSE: {mse_train:.4f}")

#testing
y_pred_test = X_test*w + b
mse_test = np.mean((y_test - y_pred_test) ** 2)

print("\nFinal Parameters:")
print(f"Weight (w): {w[0]:.4f}")
print(f"Bias (b): {b[0]:.4f}")
print(f"Test MSE: {mse_test:.4f}")

#plot
plt.figure(figsize=(12, 5))

# Scatter plot of data and regression line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color="blue", label="Data", alpha=0.6)
plt.plot(X, w * X + b, color="red", label="Regression Line")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.title("Linear Regression Results After Fixes")

# Plot of training loss over epochs
plt.subplot(1, 2, 2)
plt.plot(range(epochs), training_losses, color="green", label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.title("Training Loss Over Epochs")

plt.tight_layout()
plt.show()

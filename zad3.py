import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

years = np.array([2000, 2002, 2005, 2007, 2010])
percentages = np.array([6.5, 7.0, 7.4, 8.2, 9.0])
years_norm = (years - years.mean()) / years.std()

def gradient_descent(x, y, learning_rate, iterations):
    m, b = 0, 0
    n = len(x)

    for _ in range(iterations):
        y_predicted = m * x + b
        dm = (-2 / n) * sum(x * (y - y_predicted))
        db = (-2 / n) * sum(y - y_predicted)
        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b

learning_rate = 0.01
iterations = 1000
m, b = gradient_descent(years_norm, percentages, learning_rate, iterations)

year_12 = (12 - b) / m
year_12 = year_12 * years.std() + years.mean()

print(f"Model regresji liniowej: y = {m:.3f} * x + {b:.3f}")
print(f"Procent bezrobotnych przekroczy 12% w roku: {int(np.round(year_12))}")

fig, ax = plt.subplots()
line, = ax.plot(years, percentages, "ro")

def animate(i):
    m, b = gradient_descent(years_norm, percentages, learning_rate, i)
    y = m * years_norm + b
    line.set_ydata(y)
    return line,

ani = FuncAnimation(fig, animate, frames=iterations, interval=10, blit=True)
plt.xlabel("Lata")
plt.ylabel("Procent")
plt.title("Regresja liniowa z wykorzystaniem spadku gradientu")
plt.show()

#nowe zadania

def step(x):
    return np.where(x >= 0, 1, 0)


def perceptron_learn(X, y):
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0

    for _ in range(1000):
        for i in range(n_samples):
            z = np.dot(X[i], w) + b
            a = step(z)
            error = y[i] - a
            w += error * X[i]
            b += error

    return w, b


def perceptron_network(x1, x2):
    w_hidden = np.array([[20, -20], [-20, 20]])
    b_hidden = np.array([-10, 30])

    w_output = np.array([20, 20])
    b_output = -30

    z_hidden = np.dot(w_hidden, np.array([x1, x2])) + b_hidden
    a_hidden = step(z_hidden)

    z_output = np.dot(w_output, a_hidden) + b_output
    a_output = step(z_output)

    return a_output


# AND
print("\nAND:")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
w_and, b_and = perceptron_learn(X_and, y_and)
print("w_and = ", w_and)
print("b_and = ", b_and)

# NOT
print("\nNOT:")
X_not = np.array([[0], [1]])
y_not = np.array([1, 0])
w_not, b_not = perceptron_learn(X_not, y_not)
print("w_not = ", w_not)
print("b_not = ", b_not)

# XOR
print("\nXOR:")
print("1 XOR 1 =", perceptron_network(1, 1))
print("1 XOR 0 =", perceptron_network(1, 0))
print("0 XOR 1 =", perceptron_network(0, 1))
print("0 XOR 0 =", perceptron_network(0, 0))

#nowe zadania


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

W1 = np.array([[0.15, 0.2], [0.25, 0.3]])
b1 = np.array([0.35, 0.35])
W2 = np.array([[0.4, 0.45]])
b2 = np.array([0.6])

x = np.array([[0.6, 0.1], [0.2, 0.3]])
y = np.array([[1], [0]])

alpha = 0.1

epochs = 10000

for epoch in range(epochs):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2.T) + b2
    a2 = sigmoid(z2)

    delta2 = (a2 - y) * a2 * (1 - a2)
    delta1 = np.dot(delta2, W2) * a1 * (1 - a1)

    W2 = W2 - alpha * np.dot(a1.T, delta2)
    b2 = b2 - alpha * np.sum(delta2, axis=0, keepdims=True)
    W1 = W1 - alpha * np.dot(x.T, delta1)
    b1 = b1 - alpha * np.sum(delta1, axis=0)

print(f"{x} -> {a2}")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Data
data = """6.18;-6.92;1
-8.80;-7.34;0
0.71;5.67;1
-4.68;6.93;0
2.21;-9.48;0
6.21;1.60;1
-4.95;-6.05;0
1.99;-6.22;0
-7.60;-2.19;0
-4.86;-5.32;0
9.44;8.27;1
-5.75;0.16;0
7.48;9.05;1
0.18;-3.03;0
-2.52;5.32;0
-9.96;-3.42;0
1.22;-1.78;0
5.56;-1.04;1
-8.14;1.93;0
-9.51;7.51;0
-8.68;5.27;0
-8.86;2.54;0
-8.53;8.22;0
7.95;-4.15;1
8.74;9.85;1
2.45;-8.25;0
-3.10;8.82;0
3.57;-9.25;0
-8.21;0.07;0
-0.21;-8.35;0
3.58;-7.81;0
-9.98;-5.31;0
6.78;8.55;1
-5.50;3.99;0
6.70;-7.35;1
-0.41;-8.91;0
1.92;-7.15;0
5.30;-6.97;1
-3.10;-5.03;0
3.08;-1.94;1
5.98;2.89;1
-7.61;0.22;0
-8.94;-3.80;0
9.94;-0.21;1
8.39;2.21;1
-7.59;-8.91;0
7.22;4.61;1
5.02;-8.89;1
2.55;-4.29;0
2.02;2.94;1
-2.22;0.42;0
2.21;-3.34;0
-1.27;0.86;0
3.12;6.80;1
0.19;8.15;1
0.61;-9.64;0
-1.50;4.81;0
7.39;2.30;1
9.12;-4.75;1
-6.68;6.43;0
7.05;-5.38;1
3.31;8.77;1
-0.52;-3.01;0
4.51;-2.20;1
5.32;-7.21;1
-6.69;1.26;0
-5.83;-7.95;0
-5.02;3.04;0
8.29;-4.44;1
-5.94;4.88;0
-4.06;-1.08;0
1.66;7.83;1
1.31;-4.01;0
-2.16;6.06;0
7.32;7.64;1
1.45;-8.19;0
-9.03;-0.68;0
-2.29;4.19;0
7.84;6.73;1
-5.20;-2.85;0
-7.75;2.66;0
4.54;-7.87;1
2.72;-9.94;0
9.42;-0.53;1
-1.54;-3.49;0
8.72;-0.72;1
2.05;-6.51;0
9.40;7.49;1
5.76;-9.85;1
4.41;-4.61;1
-5.26;-4.11;0
0.34;2.13;0
8.77;8.32;1
-0.71;-0.98;0
6.27;4.49;1
5.70;-8.72;1
9.05;-8.63;1
-2.33;-1.12;0
-5.53;9.23;0"""

# Convert data to NumPy arrays
data = np.array([x.split(';') for x in data.split('\n')], dtype=float)

# Separate features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Perceptron model
perceptron = Perceptron()

# Train the model
perceptron.fit(X_train, y_train)

# Predict on the testing set
y_pred = perceptron.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("___________________________________________________________________________________________\n")
print("Your perceptron model has been trained, look below for more information.")
print("Accuracy:", accuracy)
print("Threshold:", perceptron.intercept_)
print("Weights:", perceptron.coef_)
print("\n_____________________________________________________________________________________________")

# Plot the data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')

# Plot the decision boundary and margins
w = perceptron.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]))
yy = a * xx - (perceptron.intercept_[0]) / w[1]
margin = 1 / np.sqrt(np.sum(perceptron.coef_ ** 2))
yy_down = yy - a * margin
yy_up = yy + a * margin

plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.xlabel('Feature One')
plt.ylabel('Feature Two')
plt.title('Perceptron Decision Boundary and Margins by Maseti')
plt.show()

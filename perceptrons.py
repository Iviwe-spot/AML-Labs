import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights and threshold randomly
        self.weights = np.random.uniform(-1, 1, input_size)
        self.threshold = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate

    def predict(self, x):
        # Calculate the weighted sum of inputs
        weighted_sum = np.dot(x, self.weights)

        # Apply step function as activation
        output = 1 if weighted_sum >= self.threshold else 0
        return output

    def train(self, X, y, max_epochs=100):
        # Training loop
        for epoch in range(max_epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                # Predict output
                output = self.predict(x)

                # Update weights and threshold
                self.weights += self.learning_rate * (target - output) * x
                self.threshold -= self.learning_rate * (target - output)
    
    def print_weights(self):
        print("Weights:", self.weights)
        print("Threshold:", self.threshold)

# Example usage:
# Create synthetic data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Create and train perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1)
perceptron.train(X, y)

# Print trained weights and threshold
perceptron.print_weights()

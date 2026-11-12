# Adaline

A single neuron classifier in NumPy, trained with mini-batch gradient descent.

The difference between Adaline and the perceptron is in how the weights are updated. The perceptron uses the output after the step function, so it only learns from misclassified points. Adaline uses the continuous output before thresholding, so every point contributes to the gradient and there is a real loss function to minimize.

In this implementation I used a sigmoid activation with the cross entropy loss, instead of the classic linear activation with squared error. So the update is

```
w = w - lr * (1/b) * X_b^T (sigmoid(X_b w) - y_b)
```

and the model is close to logistic regression. Predictions are 1 when the output is above 0.5.

## Test

Synthetic binary data from `make_classification` (100 samples, 2 features), 70/30 train/test split, learning rate 0.01, batch size 16, 1000 epochs.

The loss goes from about 0.72 to 0.07 and the accuracy on the test set is 100%.

## Requirements

numpy, scikit-learn (for the data, the split and `accuracy_score`)

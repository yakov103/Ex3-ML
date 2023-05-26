# Import necessary libraries
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(file_name, random_state):
    """
    Load and preprocess data from a text file.

    Parameters:
    - file_name (str): path to the text file
    - random_state (int): random state for train test split

    Returns:
    - X_train, Y_train, X_test, Y_test (tuple of np.array): split datasets
    """
    # Initialize empty lists for features and labels
    data, labels = [], []

    with open(file_name, 'r') as file:
        for line in file.readlines():
            line_data = list(map(float, line.split()))
            data.append(np.array(line_data[:-1]))
            labels.append(line_data[-1])

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Convert 0 labels to -1
    labels[labels == 0.0] = -1

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, labels, test_size=0.5, random_state=random_state)

    return X_train, Y_train, X_test, Y_test


def generate_circles(X):
    """
    Generate circles based on input data.

    Parameters:
    - X (np.array): Input data

    Returns:
    - circles (list of tuples): List of generated circles. Each circle is represented as a tuple (center, radius, direction)
    """
    circles = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if np.array_equal(X[i], X[j]):
                continue
            center = X[i]
            radius = np.linalg.norm(X[i] - X[j])
            # Create circles with both directions
            for direction in [-1, 1]:
                circles.append((center, radius, direction))
    return circles


def predict_label(circle, point):
    """
    Predict label for a given point based on a circle.

    Parameters:
    - circle (tuple): a circle as a tuple (center, radius, direction)
    - point (np.array): a data point

    Returns:
    - (int): Predicted label (-1 or 1)
    """
    center, radius, direction = circle
    if np.linalg.norm(center - point) < radius:
        return direction
    else:
        return -direction


def predict(X, circle):
    """
    Predict labels for given data based on a circle.

    Parameters:
    - X (np.array): Input data
    - circle (tuple): A circle as a tuple (center, radius, direction)

    Returns:
    - prediction (np.array): Predicted labels
    """
    prediction = np.array([predict_label(circle, point) for point in X])
    return prediction


# The remaining functions adaboost and ada_error remain the same as they do not depend on the type of hypotheses (lines or circles)

def adaboost(X, Y, num_rules, rules):
    """
    Implement the AdaBoost algorithm.

    Parameters:
    - X (np.array): Input data
    - Y (np.array): Labels
    - num_rules (int): Number of rules
    - rules (list of tuples): List of generated rules

    Returns:
    - rule_weights (np.array): Weights of the rules
    - rule_indices (list): Indices of the rules
    """
    # Initialize weights
    sample_weights = np.ones_like(Y) * (1 / len(X))

    rule_weights = []
    rule_indices = []

    for t in range(num_rules):
        errors = []

        # For each rule, compute error
        for rule in rules:
            predictions = predict(X, rule)
            error = np.dot(sample_weights, np.not_equal(predictions, Y))
            errors.append(error)

        # Get rule with minimum error
        min_error_rule_index = np.argmin(errors)
        rule_indices.append(min_error_rule_index)

        # Calculate rule weight
        min_error = errors[min_error_rule_index]
        rule_weight = 0.5 * np.log((1 - min_error) / min_error)
        rule_weights.append(rule_weight)

        # Update sample weights
        predictions = predict(X, rules[min_error_rule_index])
        sample_weights *= np.exp(-rule_weight * predictions * Y)
        sample_weights /= np.sum(sample_weights)

    return rule_weights, rule_indices


def ada_error(X, Y, rules, rule_weights, rule_indices, k):
    """
    Calculate error for AdaBoost algorithm.

    Parameters:
    - X (np.array): Input data
    - Y (np.array): Labels
    - rules (list of tuples): List of generated rules
    - rule_weights (np.array): Weights of the rules
    - rule_indices (list): Indices of the rules
    - k (int): The number of rules to use

    Returns:
    - error (float): The calculated error
    """
    # Initialize prediction array
    F = np.zeros_like(Y)

    # Compute prediction
    for i in range(k):
        F += predict(X, rules[rule_indices[i]]) * rule_weights[i]

    # Calculate error
    final_prediction = np.sign(F)
    error = np.sum(np.not_equal(Y, final_prediction))

    return error / len(Y)



if __name__ == "__main__":
    # Define constants
    n_iterations = 50
    num_circles = 8

    # Initialize error arrays
    train_errors = np.zeros(num_circles)
    test_errors = np.zeros(num_circles)

    # Main loop
    for i in range(n_iterations):
        print(f"Iteration: {i}")
        X_train, Y_train, X_test, Y_test = load_data("squares.txt", 2 * i)
        circles = generate_circles(X_train)
        circle_weights, circle_indices = adaboost(X_train, Y_train, num_circles, circles)

        for j in range(num_circles):
            train_error = ada_error(X_train, Y_train, circles, circle_weights, circle_indices, j + 1)
            test_error = ada_error(X_test, Y_test, circles, circle_weights, circle_indices, j + 1)
            train_errors[j] += train_error
            test_errors[j] += test_error

    # Compute average errors
    train_errors /= n_iterations
    test_errors /= n_iterations

    # Print results
    for i in range(num_circles):
        print(f"Average training error of circle {i}: {train_errors[i]}")
        print(f"Average test error of circle {i}: {test_errors[i]}")

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


def generate_rules(X):
    """
    Generate rules based on input data.

    Parameters:
    - X (np.array): Input data

    Returns:
    - lines (list of tuples): List of generated rules
    """
    lines = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if np.array_equal(X[i], X[j]) or X[j][0] - X[i][0] == 0 or X[j][1] - X[i][1] == 0:
                continue
            slope = (X[j][1] - X[i][1]) / (X[j][0] - X[i][0])
            intercept = X[i][1] - slope * X[i][0]
            lines.append((slope, intercept))
    return lines


def predict_label(rule, point):
    """
    Predict label for a given point based on a rule.

    Parameters:
    - rule (tuple): a rule as a tuple (slope, intercept)
    - point (np.array): a data point

    Returns:
    - (int): Predicted label (-1 or 1)
    """
    pred = rule[0] * point[0] + rule[1]
    return 1 if pred > point[1] else -1


def is_rule_correct(rule, point, label):
    """
    Check if a rule predicts the correct label for a point.

    Parameters:
    - rule (tuple): a rule as a tuple (slope, intercept)
    - point (np.array): a data point
    - label (int): the correct label of the point

    Returns:
    - (int): 1 if the rule is wrong, 0 otherwise
    """
    return 1 if predict_label(rule, point) != label else 0


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


def predict(X, rule):
    """
    Predict labels for given data based on a rule.

    Parameters:
    - X (np.array): Input data
    - rule (tuple): A rule as a tuple (slope, intercept)

    Returns:
    - prediction (np.array): Predicted labels
    """
    mask = X[:, 1] < (rule[0] * X[:, 0] + rule[1])
    prediction = np.zeros_like(X[:, 0]) - 1
    prediction[mask] = 1
    return prediction

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
    num_rules = 8

    # Initialize error arrays
    train_errors = np.zeros(num_rules)
    test_errors = np.zeros(num_rules)

    # Main loop
    for i in range(n_iterations):
        print(f"Iteration: {i}")
        X_train, Y_train, X_test, Y_test = load_data("squares.txt", 2 * i)
        rules = generate_rules(X_train)
        rule_weights, rule_indices = adaboost(X_train, Y_train, num_rules, rules)

        for j in range(num_rules):
            train_error = ada_error(X_train, Y_train, rules, rule_weights, rule_indices, j + 1)
            test_error = ada_error(X_test, Y_test, rules, rule_weights, rule_indices, j + 1)
            train_errors[j] += train_error
            test_errors[j] += test_error

    # Compute average errors
    train_errors /= n_iterations
    test_errors /= n_iterations

    # Print results
    for i in range(num_rules):
        print(f"Empirical error mean of line rule H-{i} is : {train_errors[i]}")
        print(f"Error mean of line rule H-{i} is: {test_errors[i]}")

import numpy as np

def split_data(X, y, test_size, val_size):
    """
    Splits the data into train, validation and test sets.
    
    :param X: numpy array, feature matrix
    :param y: numpy array or list, target values
    :param test_size: float, proportion of the dataset to include in the test split (0.0 to 1.0)
    :param val_size: float, proportion of the dataset to include in the validation split (0.0 to 1.0)
    
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Get unique classes
    classes = np.unique(y)
    
    # Get the number of samples per class for the test and validation sets
    n_test = int(min(len(y[y == c]) for c in classes) * test_size)
    n_val = int(min(len(y[y == c]) for c in classes) * val_size)
    
    # Initialize empty lists for the final train, validation and test sets
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    # For each class, split the data into train, validation and test sets
    for c in classes:
        # Get the data for this class
        X_c = X[y == c]
        y_c = y[y == c]
        
        # Get the test, validation and training data for this class
        X_test_c = X_c[:n_test]
        X_val_c = X_c[n_test:n_test+n_val]
        X_train_c = X_c[n_test+n_val:]
        
        y_test_c = y_c[:n_test]
        y_val_c = y_c[n_test:n_test+n_val]
        y_train_c = y_c[n_test+n_val:]
        
        # Append the data to the final train, validation and test sets
        X_train.append(X_train_c)
        X_val.append(X_val_c)
        X_test.append(X_test_c)
        y_train.append(y_train_c)
        y_val.append(y_val_c)
        y_test.append(y_test_c)
        
    # Concatenate the data
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)
    y_test = np.concatenate(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Example usage:
# X, y = ...
# test_size = 0.2
# val_size = 0.2
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size, val_size)

"""functions for implementing logistic regression"""
import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z) applied element wise if z is a numpy array
    """
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    """
    Initialize parameters (w and b) of logistic regression with zeros
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    return np.zeros((dim, 1)), 0

def propagate(w, b, X, Y):
    """
    Compute the cost function and the gradients given parameters w and b,
    training set X and labels Y
    
    Arguments:
    w -- numpy vector of dimension (X.shape[0], 1)
    b -- scalar corresponding to the bias
    X -- numpy array of dimension (n, m)
    Y -- numpy vector of dimension (1, m)
    
    Returns:
    grads -- dictionary with keys dw and db, corresponding to the gradients
    cost -- the cost function
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(- (Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    """
    Run the gradient descent algorithm num_iterations times given the initial parameters,
    the training set and the labels, and return the learned parameters
    
    Arguments:
    w -- numpy vector of dimension (X.shape[0], 1)
    b -- scalar corresponding to the bias
    X -- numpy array of dimension (n, m)
    Y -- numpy vector of dimension (1, m)
    num_iterations -- integer for the number of iterations of gradient descent
    learning_rate -- the learning rate of the gradient descent algorithm
    print_cost -- if set to True, the costs will be printed every 100 iterations
    
    Returns:
    
    params -- dictionary with keys w and b, corresponding to the learned parameters
    grads -- dictionary with keys dw and db, corresponding to the gradients
    costs -- array of the cost every 100 iteration for plotting the learning curve
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i+1, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def predict(w, b, X):
    """
    Compute a vector of predictions on X given the learned parameters w and b
    
    Arguments:
    w -- numpy vector of dimension (X.shape[0], 1) corresponding to the learned parameters
    b -- scalar corresponding to the learned bias
    X -- numpy array
    
    Returns:
    Y_predictions: numpy array of dimension (1, X.shape[1])
    """
    Y_predictions = np.zeros((1, X.shape[1]))
    A = sigmoid(np.dot(w.T, X) + b)
    Y_predictions[:] = A > 0.5
    return Y_predictions

def model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.005, print_cost=False):
    """
    Builds a logistic regression model
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (n_train, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (n_test, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_predictions_train = predict(w, b, X_train)
    Y_predictions_test = predict(w, b, X_test)
    d = {"w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "costs": costs,
         "Y_predictions_train": Y_predictions_train,
         "Y_predictions_test": Y_predictions_test}
    return d

    
    
    
    
    
    
    
    
    
    
    
    
    
    

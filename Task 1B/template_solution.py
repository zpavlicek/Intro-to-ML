# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component in a given row of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_input: matrix of floats: dim = (700,21), transformed input with 21 features
    """
    X_input = np.zeros((700, 21))
    # TODO: Enter your code here
    for column in range (21):
        if column <= 4:
            X_input[:,column] = X[:,column]
        elif column <= 9:
            X_input[:,column] = X[:,column%5]**2
        elif column <= 14:
            X_input[:,column] = np.exp(X[:,column%5])
        elif column <= 19:
            X_input[:,column] = np.cos(X[:,column%5])
        else:
            X_input[:,column] = np.ones(700)
    assert X_input.shape == (700, 21)
    print(X_input)
    return X_input



def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels

    Returns
    ----------
    weights: array of floats: dim = (21,), optimal parameters of linear regression
    """
    weights = np.zeros((21,))
    X_input = transform_data(X)
    # TODO: Enter your code here
    #weights = np.linalg.inv(X_input.T @ X_input) @ X_input.T @ y
    
    epsilon = 0.0001
    print("Shape of X:", X_input.shape)
    print("Shape of w.T:", weights.T.shape)
    print("Shape of y:", y.shape)

    L = np.mean((y- X_input @ weights.T)**2)
    gradL = (2 / 700) * (X_input.T @ X_input @ weights - X_input.T @ y)
    
    weights_old = weights
    weights = weights - 0.1*gradL
    
    L_old = L
    L = np.mean((y- X_input @ weights_old.T)**2)
    
    while abs(L - L_old) > epsilon:
        gradL = (2 / 700) * (X_input.T @ X_input @ weights - X_input.T @ y)
        weights = weights - 0.1*gradL
        L_old = L
        L = np.mean((y- X_input @ weights_old.T)**2)
        
    assert weights.shape == (21,)
    return weights


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())
   

    X = data.to_numpy()
    transform_data(X)
    np.savetxt("matrix.csv", transform_data(X), delimiter=",", fmt="%.4f")
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")

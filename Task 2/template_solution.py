
# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel, Product, PairwiseKernel, Exponentiation, CompoundKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    X_train = train_df.drop(columns=['price_CHF', "season"])
    print(X_train)
   
    X_train = X_train.to_numpy()
    np.savetxt("data_matrix.csv", X_train, delimiter=",", fmt="%.4f")
    
    y_train = train_df["price_CHF"].to_numpy()
    np.savetxt("y_train.csv", y_train, delimiter=",", fmt="%.4f")
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    X_test = test_df.drop(columns=["season"])
    X_test = X_test.to_numpy()
    
    #print("Test data:")
    #print(test_df.shape)
    #print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    
    for column in range (X_train.shape[1]):
        for row in range (X_train.shape[0]):
    
            if np.isnan(X_train[row, column]): 
                season_index = row % 4  # 0: spring, 1: summer, etc.
                valid_values = []

                # Look before
                count = 0
                i = row - 1
                while i >= 0 and count < 1:
                    if i % 4 == season_index and not np.isnan(X_train[i, column]):
                        valid_values.append(X_train[i, column])
                        count += 1
                    i -= 1

                # Look after
                count = 0
                i = row + 1
                while i < X_train.shape[0] and count < 1:
                    if i % 4 == season_index and not np.isnan(X_train[i, column]):
                        valid_values.append(X_train[i, column])
                        count += 1
                    i += 1

                if valid_values:
                    X_train[row, column] = np.mean(valid_values)
            
    np.savetxt("data_matrix_mean.csv", X_train, delimiter=",", fmt="%.4f")
    
    for index in range(y_train.shape[0]):
        if np.isnan(y_train[index]):
            season_index = index % 4  # Group by season pattern

            valid_values = []

            # Look 5 values before (same season)
            i = index - 1
            count = 0
            while i >= 0 and count < 1:
                if i % 4 == season_index and not np.isnan(y_train[i]):
                    valid_values.append(y_train[i])
                    count += 1
                i -= 1

            # Look 5 values after (same season)
            i = index + 1
            count = 0
            while i < y_train.shape[0] and count < 1:
                if i % 4 == season_index and not np.isnan(y_train[i]):
                    valid_values.append(y_train[i])
                    count += 1
                i += 1

            # Replace NaN with mean of same-season values
            if valid_values:
                y_train[index] = np.mean(valid_values)
        
    np.savetxt("y_train_mean.csv", y_train, delimiter=",", fmt="%.4f")
    
    for column in range (X_test.shape[1]):
        for row in range (X_test.shape[0]):
    
            if np.isnan(X_test[row, column]): 
                season_index = row % 4  # 0: spring, 1: summer, etc.
                valid_values = []

                # Look before
                count = 0
                i = row - 1
                while i >= 0 and count < 1:
                    if i % 4 == season_index and not np.isnan(X_test[i, column]):
                        valid_values.append(X_test[i, column])
                        count += 1
                    i -= 1

                # Look after
                count = 0
                i = row + 1
                while i < X_test.shape[0] and count < 1:
                    if i % 4 == season_index and not np.isnan(X_test[i, column]):
                        valid_values.append(X_test[i, column])
                        count += 1
                    i += 1

                if valid_values:
                    X_test[row, column] = np.mean(valid_values)
            
    np.savetxt("test_matrix_mean.csv", X_test, delimiter=",", fmt="%.4f")
    

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._model = None
        self._scaler = MinMaxScaler() 

    def cross_validate_and_select_kernel(self, X: np.ndarray, y: np.ndarray, k=3):
        kernels = [DotProduct(), RBF(), RBF() + RationalQuadratic() * DotProduct(), Matern(), WhiteKernel(), RationalQuadratic(), RBF() * RationalQuadratic() * DotProduct()]
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_values_per_kernel = {str(kernel): [] for kernel in kernels}

        for kernel in kernels:
            print(f'\nTesting kernel: {kernel}')
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                gpr = GaussianProcessRegressor(kernel=kernel, normalize_y = True)
                gpr.fit(X_scaled, y_train)
                y_pred_train = gpr.predict(X_val_scaled)

                mse = np.mean((y_val-y_pred_train)**2)
                mse_values_per_kernel[str(kernel)].append(mse)
                print(f'Fold MSE: {mse:.4f}')

            avg_mse = np.mean(mse_values_per_kernel[str(kernel)])
            print(f'Average MSE for kernel {kernel}: {avg_mse:.4f}')

        best_kernel_str = min(mse_values_per_kernel, key=lambda k: np.mean(mse_values_per_kernel[k]))
        best_kernel_mse = np.mean(mse_values_per_kernel[best_kernel_str])
        print(f'\nBest kernel based on average MSE: {best_kernel_str}, with Average MSE: {best_kernel_mse:.4f}')

        # Create best kernel object again (string to object workaround)
        for kernel in kernels:
            if str(kernel) == best_kernel_str:
                return kernel

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self._x_train = X_train
        self._y_train = y_train
        self._scaler.fit(X_train)
        X_scaled = self._scaler.transform(X_train)
        best_kernel = self.cross_validate_and_select_kernel(X_scaled, y_train)
        print(f"\nTraining final model with best kernel: {best_kernel}")
        self._model = GaussianProcessRegressor(kernel=best_kernel)
        self._model.fit(X_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X_test)
        y_pred = self._model.predict(X_scaled)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred   


def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - y_mean) ** 2)  # Total sum of squares

    r2 = 1 - (ss_res / ss_tot)
    return r2
    
# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    model = Model()
    # Use this function for training the model
    model.train(X_train=X_train, y_train=y_train)
    # Use this function for inferece
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
    

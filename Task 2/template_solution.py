import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.ensemble import ExtraTreesRegressor


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
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    # Drop rows where the price in CHF is NaN
    train_df = train_df.dropna(subset=["price_CHF"]).reset_index(drop=True)

    y_train = train_df["price_CHF"].to_numpy()  #Select the prices in CHF and converts them into a NumPy array
    train_features = train_df.drop(columns=["price_CHF"]) #Select all the other colomns that we will use as inputs
    test_features = test_df.copy()

    # We have two feature types: the seasons, and the prices for different countries
    categorical_features = ["season"]
    numerical_features = [col for col in train_features.columns if col != "season"]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)), #Missing values in column estimated based on other features in the same row using ExtraTreeRegressor, that learns patterns in the data and predicts the missing value. 
        ("scaler", MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown='ignore')) #We create a new column for each season
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Fit-transform the data
    X_train = preprocessor.fit_transform(train_features)
    X_test = preprocessor.transform(test_features)
    
    #Save the preprocessed data to CSV
    pd.DataFrame(X_train).to_csv("X_train_preprocessed.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test_preprocessed.csv", index=False)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test #X_train: processed training inputs, #y_train: cleaned target values (removed the row with NANs), X_test: processed test inputs


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._model = None  # Placeholder for the final trained GPR model

    def _cross_validate_and_select_kernel(self, X, y, k=5):
        """
        Try different Gaussian Process kernels and return the one with the lowest
        average validation MSE across k folds.
        Also prints the MSE for each fold and the average per kernel.
        """
        kernels = [
            DotProduct(),
            RBF(),
            RBF() + RationalQuadratic() * DotProduct(),
            Matern(),
            WhiteKernel(),
            RationalQuadratic(),
            RationalQuadratic() +  WhiteKernel(),
            RBF() * RationalQuadratic() * DotProduct(),
            RBF() + WhiteKernel()
        ]

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_values_per_kernel = {str(kernel): [] for kernel in kernels}

        for kernel in kernels:
            print(f"\nTesting kernel: {kernel}")
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
                X_train_cv, X_val = X[train_idx], X[val_idx]
                y_train_cv, y_val = y[train_idx], y[val_idx]

                gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                gpr.fit(X_train_cv, y_train_cv)
                y_pred_val = gpr.predict(X_val)

                mse = np.mean((y_val - y_pred_val) ** 2)
                mse_values_per_kernel[str(kernel)].append(mse)
                print(f"  Fold {fold_idx} MSE: {mse:.4f}")

            avg_mse = np.mean(mse_values_per_kernel[str(kernel)])
            print(f"Average MSE for kernel {kernel}: {avg_mse:.4f}")

        best_kernel_str = min(mse_values_per_kernel, key=lambda k: np.mean(mse_values_per_kernel[k]))
        best_kernel_mse = np.mean(mse_values_per_kernel[best_kernel_str])

        print(f"\nBest kernel selected: {best_kernel_str}")
        print(f"Average MSE of selected kernel: {best_kernel_mse:.4f}")

        for kernel in kernels:
            if str(kernel) == best_kernel_str:
                return kernel

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

        best_kernel = self._cross_validate_and_select_kernel(X_train, y_train)
        self._model = GaussianProcessRegressor(kernel=best_kernel, normalize_y=True)
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = np.zeros(X_test.shape[0])
        # TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self._model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


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

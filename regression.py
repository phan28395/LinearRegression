# write your code here
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression:
    def __init__(self, *, fit_intercept):
        # The fit_intercept value will determine whether our regression model have the intercept value or not
        self.fit_intercept = fit_intercept
        self.coefficient = []
        self.intercept = 0

    def fit(self, x_df, y_series):
        """This function will calculate the coefficient"""

        x_numpy = x_df.to_numpy()
        y_numpy = np.array([y for y in y_series])
        x_t = np.transpose(x_numpy)
        x_t_x = np.dot(x_t, x_numpy)
        x_t_x_inverse = np.linalg.inv(x_t_x)
        xt_x_inverse_xt = np.dot(x_t_x_inverse, x_t)
        return np.dot(xt_x_inverse_xt, y_numpy)

    def predict(self, x_df, y_series):
        """This function will calculate the predict y (This is the
        value after you have the coefficient and then apply it in to
        the linear regression model ). Base on values of matrix X and the calculated coefficient in function fit() """
        x_numpy = x_df.to_numpy()
        coefficient = self.fit(x_df, y_series)
        return np.dot(x_numpy, coefficient)

    def r2_score(self, y_series, x_df):
        """This function will calculate the r^2-score
        which is the coefficient of determination, the higher
        the score, the better the linear regression model.
        """
        y_numpy = np.array([y for y in y_series])
        y_hat_numpy = self.predict(x_df, y_series)
        y_numpy_mean = y_numpy.mean()
        subtract_sigma_numerator = 0
        subtract_sigma_denominator = 0
        for i in range(len(y_series)):
            subtract_sigma_numerator += (y_numpy[i] - y_hat_numpy[i]) ** 2
        for i in range(len(y_series)):
            subtract_sigma_denominator += (y_numpy[i] - y_numpy_mean) ** 2
        return 1 - (subtract_sigma_numerator / subtract_sigma_denominator)

    def rmse(self, y_series, x_df):
        """This function will calculate the metric RMSE, which is the square root of the
        mean squared error."""
        y_numpy = np.array([y for y in y_series])
        y_hat_numpy = self.predict(x_df, y_series)
        subtract_sigma = 0
        for i in range(len(y_series)):
            subtract_sigma += (y_numpy[i] - y_hat_numpy[i]) ** 2
        mse = (1 / (len(y_series))) * subtract_sigma
        return (mse) ** (1 / 2)


class SklearnModell:

    df = pd.read_csv('data_stage4.csv')
    X = df.iloc[:, :3]
    Y = df.y[:]
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)
    intercept = model.intercept_
    coefficient = model.coef_
    Y_predict = model.predict(X)

    r2_score = r2_score(Y, Y_predict)
    mse_score = mean_squared_error(Y, Y_predict)
    rmse_score = mse_score ** (1/2)




def main():
    df = pd.read_csv('data_stage4.csv')
    X = df.iloc[:, :3]
    Y = df.y[:]
    model = LinearRegression(fit_intercept=True)
    model.fit(X, Y)
    regression = CustomLinearRegression(fit_intercept=True)
    regression_sklearn = SklearnModell()

    if regression.fit_intercept:
        intercept = []
        for i in range(len(X['f1'])):
            intercept.append(1)
        X['wo'] = intercept

    for element in regression.fit(X, Y):
        regression.coefficient.append(element)
    if regression.fit_intercept:
        regression.intercept = regression.coefficient[len(X.columns) - 1]
        regression.coefficient.pop()

    output = {'Intercept': regression.intercept - regression_sklearn.intercept,
              'Coefficient': np.array(regression.coefficient) - np.array(regression_sklearn.coefficient),
              'R2': regression.r2_score(Y, X) - regression_sklearn.r2_score,
              'RMSE': regression.rmse(Y, X) - regression_sklearn.rmse_score}
    print(output)

if __name__ == "__main__":
    main()

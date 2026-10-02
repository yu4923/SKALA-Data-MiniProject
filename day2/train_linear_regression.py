from sklearn.linear_model import LinearRegression
from training import train

PARAMS = {'fit_intercept': False, 'positive': True}


if __name__ == '__main__':
    train('linear_regression',
          lambda columns: LinearRegression(**PARAMS),
          tuned_params=PARAMS)

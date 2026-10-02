from sklearn.linear_model import LinearRegression
from training import train


if __name__ == '__main__':
    train('linear_regression', lambda columns: LinearRegression())

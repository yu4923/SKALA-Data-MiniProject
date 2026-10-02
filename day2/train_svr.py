from sklearn.svm import SVR
from training import train

PARAMS = {'C': 10, 'epsilon': 1, 'kernel': 'linear'}


if __name__ == '__main__':
    train('svr', lambda columns: SVR(**PARAMS), tuned_params=PARAMS)

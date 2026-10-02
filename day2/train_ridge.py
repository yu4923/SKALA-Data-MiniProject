from sklearn.linear_model import Ridge
from training import train

PARAMS = {'alpha': 10}


if __name__ == '__main__':
    train('ridge',
          lambda columns: Ridge(**PARAMS),
          tuned_params=PARAMS)

from sklearn.ensemble import GradientBoostingRegressor
from training import train

PARAMS = {'learning_rate': 0.1, 'max_depth': 1, 'min_samples_leaf': 1, 'n_estimators': 300}


if __name__ == '__main__':
    train('gradient_boosting',
          lambda columns: GradientBoostingRegressor(random_state=42, **PARAMS),
          tuned_params=PARAMS)

from sklearn.ensemble import ExtraTreesRegressor
from training import train

PARAMS = {'max_depth': 5, 'max_features': 0.7, 'min_samples_leaf': 1, 'n_estimators': 300}


if __name__ == '__main__':
    train('extra_trees',
          lambda columns: ExtraTreesRegressor(random_state=42, n_jobs=-1, **PARAMS),
          tuned_params=PARAMS)

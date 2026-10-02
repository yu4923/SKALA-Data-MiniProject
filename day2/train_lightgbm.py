from lightgbm import LGBMRegressor
from training import train

PARAMS = {'num_leaves': 7, 'n_estimators': 300, 'min_child_samples': 2, 'max_depth': -1, 'learning_rate': 0.1}


if __name__ == '__main__':
    train('lightgbm',
          lambda columns: LGBMRegressor(random_state=42, n_jobs=4, verbosity=-1, **PARAMS),
          tuned_params=PARAMS)

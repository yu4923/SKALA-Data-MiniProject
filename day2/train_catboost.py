from catboost import CatBoostRegressor
from training import train

PARAMS = {'depth': 7, 'iterations': 600, 'l2_leaf_reg': 10, 'learning_rate': 0.03}


if __name__ == '__main__':
    train('catboost',
          lambda columns: CatBoostRegressor(loss_function='RMSE', boosting_type='Ordered', random_seed=42, verbose=False, allow_writing_files=False, thread_count=4, **PARAMS), categorical=True,
          tuned_params=PARAMS)

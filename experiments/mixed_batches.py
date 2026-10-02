from pathlib import Path
from importlib import import_module
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'day2'))
from features import build_features, input_columns, common
from training import make_pipeline


def filter_qd(frame, train, data_dir):
    raw = common.get_context(data_dir).df
    raw = raw[raw['cycle'].between(1, 100)].copy()
    raw['QD'] = raw['QD'].replace([np.inf, -np.inf], np.nan)
    early = raw[raw['cycle'].between(1, 5)].merge(
        train[['batch_id', 'cell_id']], on=['batch_id', 'cell_id'], validate='many_to_one')
    nominal = early.groupby(['batch_id', 'cell_id'])['QD'].median().median()
    raw.loc[~raw['QD'].between(nominal * 0.8, nominal * 1.2), 'QD'] = np.nan
    values = raw.groupby(['batch_id', 'cell_id'])['QD'].agg(
        mean_QD='mean', std_QD='std').reset_index()
    return frame.drop(columns=['mean_QD', 'std_QD']).merge(
        values, on=['batch_id', 'cell_id'], how='left', validate='one_to_one')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=common.DATA_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = build_features(args.data_dir)
    frame = frame[frame['cycle_life'].notna()].copy()
    short = frame[(frame['batch_id'] == 2) & (frame['cycle_life'] < 500)]
    short = short.sample(frac=1, random_state=args.seed)
    cut = len(short) // 2
    added_train, added_test = short.iloc[:cut], short.iloc[cut:]
    batch1 = frame[frame['batch_id'] == 1]
    mixed_train = pd.concat([batch1, added_train], ignore_index=True)
    test = pd.concat([frame[frame['batch_id'] == 3], added_test], ignore_index=True)
    assert not set(map(tuple, mixed_train[['batch_id', 'cell_id']].to_numpy())) & set(
        map(tuple, test[['batch_id', 'cell_id']].to_numpy()))
    print(f'Train: Batch 1 {len(batch1)} + Batch 2 {len(added_train)}')
    print(f'Test: Batch 3 {len(test) - len(added_test)} + Batch 2 {len(added_test)}')
    print('Batch 2 train cell_id:', added_train['cell_id'].tolist())
    print('Batch 2 test cell_id:', added_test['cell_id'].tolist())
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(),
        'svr': SVR(),
        'random_forest': RandomForestRegressor(random_state=42, n_jobs=4),
        'extra_trees': ExtraTreesRegressor(random_state=42, n_jobs=4),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'catboost': CatBoostRegressor(loss_function='RMSE', boosting_type='Ordered',
            random_seed=42, verbose=False, allow_writing_files=False, thread_count=4),
        'xgboost': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4),
        'lightgbm': LGBMRegressor(random_state=42, n_jobs=4, verbosity=-1),
    }
    columns = input_columns(frame)
    results = []
    for setting, train in [('Batch1', batch1), ('Batch1+short', mixed_train)]:
        filtered = filter_qd(frame, train, args.data_dir)
        train_data = train[['batch_id', 'cell_id']].merge(filtered, on=['batch_id', 'cell_id'])
        test_data = test[['batch_id', 'cell_id']].merge(filtered, on=['batch_id', 'cell_id'])
        for name, estimator in models.items():
            params = import_module('train_' + name).PARAMS
            estimator = clone(estimator).set_params(**params)
            categorical = name == 'catboost'
            pipeline = make_pipeline(estimator, columns, categorical)
            fit_params = {'model__cat_features': [len(columns) - 1]} if categorical else {}
            pipeline.fit(train_data[columns], train_data['cycle_life'], **fit_params)
            prediction = pipeline.predict(test_data[columns])
            b3 = test_data['batch_id'] == 3
            b2 = test_data['batch_id'] == 2
            results.append({
                'model': name, 'train': setting,
                'Test MAPE (%)': mean_absolute_percentage_error(test_data['cycle_life'], prediction) * 100,
                'Batch3 MAPE (%)': mean_absolute_percentage_error(test_data.loc[b3, 'cycle_life'], prediction[b3]) * 100,
                'Batch2 short MAPE (%)': mean_absolute_percentage_error(test_data.loc[b2, 'cycle_life'], prediction[b2]) * 100,
                'MAE': mean_absolute_error(test_data['cycle_life'], prediction),
            })
    print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == '__main__':
    main()

from pathlib import Path
from importlib import import_module
import sys
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'day1'))
common = import_module('00_data')
read_delta_q = import_module('03_delta_q').read_delta_q
build_charging_features = import_module('04_charging').build_charging_features


def build_features(data_dir=None):
    context = common.get_context(data_dir or common.DATA_DIR)
    columns = ['QD', 'QC', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']
    early = context.df[context.df['cycle'].between(1, 100)].copy()
    early[columns] = early[columns].replace([np.inf, -np.inf], np.nan)
    features = early.groupby(['batch_id', 'cell_id']).agg(
        mean_QD=('QD', 'mean'), std_QD=('QD', 'std'), mean_QC=('QC', 'mean'),
        mean_IR=('IR', 'mean'), mean_Tmax=('Tmax', 'mean'),
        mean_Tavg=('Tavg', 'mean'), mean_Tmin=('Tmin', 'mean'),
        mean_chargetime=('chargetime', 'mean'),
    ).reset_index()
    features = context.cycle_life_df.merge(features, on=['batch_id', 'cell_id'],
                                          how='left', validate='one_to_one')
    delta_features = []
    for batch_id in context.batch_ids:
        with h5py.File(context.data_dir / context.batch_names[batch_id], 'r') as mat_file:
            for cell_id in context.cycle_life_df.loc[context.cycle_life_df['batch_id'] == batch_id, 'cell_id']:
                values = read_delta_q(mat_file, int(cell_id))
                if values is None:
                    continue
                _, q10, q100 = values
                delta = q100 - q10
                delta_features.append({'batch_id': batch_id, 'cell_id': cell_id,
                                       'delta_Q_var': np.var(delta), 'delta_Q_min': np.min(delta)})
    delta_features = pd.DataFrame(delta_features,
        columns=['batch_id', 'cell_id', 'delta_Q_var', 'delta_Q_min'])
    features = features.merge(delta_features, on=['batch_id', 'cell_id'], how='left', validate='one_to_one')
    current = build_charging_features(context)
    features = features.merge(
        current[['batch_id', 'cell_id', 'mean_current_A', 'std_current_A', 'p95_current_A']],
        on=['batch_id', 'cell_id'], how='left', validate='one_to_one')
    features['charging_policy'] = features['charging_policy'].fillna('unknown').astype(str)
    numerical = features.select_dtypes(include='number').columns
    features[numerical] = features[numerical].replace([np.inf, -np.inf], np.nan)
    return features


def input_columns(features):
    return [c for c in features.columns if c not in ['batch_id', 'cell_id', 'source_file', 'cycle_life']]


if __name__ == '__main__':
    frame = build_features()
    print(frame.head().to_string(index=False))
    print(frame.groupby('batch_id')['cycle_life'].agg(['size', 'count']))

from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

common = import_module('00_data')
display = common.display

read_delta_q = import_module('03_delta_q').read_delta_q
build_charging_features = import_module('04_charging').build_charging_features

def main(context=None):
    context = context or common.get_context()
    df = context.df
    DATA_DIR = context.data_dir
    batch_ids, batch_names = context.batch_ids, context.batch_names
    cycle_life_df, life = context.cycle_life_df, context.life
    norm, cmap = context.norm, context.cmap

    charging_features = build_charging_features(context)

    # 초기 사이클 Feature와 Cycle Life 상관관계
    summary_features = ['QD', 'QC', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']
    early = df[df['cycle'].between(1, 100)].copy()
    early[summary_features] = early[summary_features].replace([np.inf, -np.inf], np.nan)
    early_features = early.groupby(['batch_id', 'cell_id']).agg(
        mean_QD=('QD', 'mean'), std_QD=('QD', 'std'), mean_QC=('QC', 'mean'),
        mean_IR=('IR', 'mean'), mean_Tmax=('Tmax', 'mean'),
        mean_Tavg=('Tavg', 'mean'), mean_Tmin=('Tmin', 'mean'),
        mean_chargetime=('chargetime', 'mean'), early_cycles=('cycle', 'nunique'),
    ).reset_index()
    early_features = early_features.merge(
        cycle_life_df[['batch_id', 'cell_id', 'cycle_life']],
        on=['batch_id', 'cell_id'], validate='one_to_one',
    )

    # ΔQ(V) 분산 / 최솟값
    delta_features = []
    for batch_id in batch_ids:
        with h5py.File(DATA_DIR / batch_names[batch_id], 'r') as mat_file:
            for cell_id in cycle_life_df.loc[cycle_life_df['batch_id'] == batch_id, 'cell_id']:
                values = read_delta_q(mat_file, int(cell_id))
                if values is None:
                    continue
                voltage, q10, q100 = values
                delta_q = q100 - q10
                delta_features.append({'batch_id': batch_id, 'cell_id': cell_id,
                                       'delta_Q_var': np.var(delta_q), 'delta_Q_min': np.min(delta_q)})
    early_features = early_features.merge(pd.DataFrame(delta_features),
        on=['batch_id', 'cell_id'], how='left', validate='one_to_one')
    early_features = early_features.merge(
        charging_features[['batch_id', 'cell_id', 'mean_current_A', 'std_current_A', 'p95_current_A']],
        on=['batch_id', 'cell_id'], how='left', validate='one_to_one',
    )
    feature_columns = [column for column in early_features.columns
                       if column not in ['batch_id', 'cell_id', 'cycle_life', 'early_cycles']]
    correlation_records = []
    correlation_matrices = {}

    for batch_id in batch_ids:
        sub = early_features[early_features['batch_id'] == batch_id]
        sub = sub.dropna(subset=['cycle_life'])
        matrix = sub[feature_columns + ['cycle_life']].corr()
        correlation_matrices[batch_id] = matrix
        for feature in feature_columns:
            pairs = sub[[feature, 'cycle_life']].dropna()
            correlation_records.append({
                'batch_id': batch_id, 'Feature': feature, '셀 수': len(pairs),
                'Pearson r': pairs[feature].corr(pairs['cycle_life']) if len(pairs) >= 3 else np.nan,
                'Spearman r': pairs[feature].rank().corr(pairs['cycle_life'].rank()) if len(pairs) >= 3 else np.nan,
            })

        fig, ax = plt.subplots(figsize=(12, 10), layout='constrained')
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(matrix)))
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_xticklabels(matrix.columns, rotation=60, ha='right', fontsize=10)
        ax.set_yticklabels(matrix.index, fontsize=10)
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                value = matrix.iloc[row, col]
                text = f'{value:.2f}' if np.isfinite(value) else '-'
                ax.text(col, row, text, ha='center', va='center', fontsize=8,
                        color='white' if np.isfinite(value) and abs(value) > 0.6 else 'black')
        ax.set_title(f'Batch {batch_id} - Early Cycle Feature Correlation')
        fig.colorbar(im, ax=ax, label='Pearson r')
        plt.show()

    life_correlation = pd.DataFrame(correlation_records)

    # 가장 강한 관계
    for batch_id in batch_ids:
        sub = life_correlation[life_correlation['batch_id'] == batch_id].copy()
        sub['|Pearson r|'] = sub['Pearson r'].abs()
        sub = sub.sort_values('|Pearson r|', ascending=False)
        print(f'Batch {batch_id} - 수명과의 상관관계')
        display(sub.drop(columns='batch_id').round(3))
        valid = sub.dropna(subset=['Pearson r'])
        if not valid.empty:
            best = valid.iloc[0]
            print(f"가장 강한 선형 관계: {best['Feature']} (r={best['Pearson r']:.3f})")

    # 다중공선성 후보
    threshold = 0.8
    pair_records = []
    for batch_id in batch_ids:
        matrix = correlation_matrices[batch_id].loc[feature_columns, feature_columns]
        for i, feature1 in enumerate(feature_columns):
            for feature2 in feature_columns[i+1:]:
                value = matrix.loc[feature1, feature2]
                if np.isfinite(value) and abs(value) >= threshold:
                    sub = early_features[early_features['batch_id'] == batch_id]
                    count = len(sub.dropna(subset=['cycle_life', feature1, feature2]))
                    pair_records.append({'batch_id': batch_id, 'Feature 1': feature1,
                                         'Feature 2': feature2, '셀 수': count,
                                         'Pearson r': value, '|r|': abs(value)})
    multicollinearity_pairs = pd.DataFrame(pair_records,
        columns=['batch_id', 'Feature 1', 'Feature 2', '셀 수', 'Pearson r', '|r|'])
    for batch_id in batch_ids:
        print(f'Batch {batch_id} - |r| >= {threshold}')
        sub = multicollinearity_pairs[multicollinearity_pairs['batch_id'] == batch_id]
        display(sub.drop(columns='batch_id').sort_values('|r|', ascending=False).round(3))

if __name__ == '__main__':
    common.run_cli(main)

from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

common = import_module('00_data')
display = common.display


def main(context=None):
    context = context or common.get_context()
    df = context.df
    DATA_DIR = context.data_dir
    batch_ids, batch_names = context.batch_ids, context.batch_names
    cycle_life_df, life = context.cycle_life_df, context.life
    norm, cmap = context.norm, context.cmap

    # 사이클별 QD 추이
    fig, axes = plt.subplots(1, len(batch_ids), figsize=(7 * len(batch_ids), 6),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    for ax, batch_id in zip(axes.flat, batch_ids):
        sub_batch = df[df['batch_id'] == batch_id]
        for _, sub in sub_batch.groupby('cell_id', sort=False):
            sub = sub.sort_values('cycle')
            life_value = sub['cycle_life'].iloc[0]
            color = cmap(norm(life_value)) if np.isfinite(life_value) else 'gray'
            ax.plot(sub['cycle'], sub['QD'], color=color, linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('QD (Ah)')
        ax.set_title(f'Batch {batch_id} - QD Curve')

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label='Cycle Life')
    plt.show()

    # Outlier 필터링 후 열화 곡선

    # 용량 범위
    nominal = df[df['cycle'] <= 5].groupby(['batch_id', 'cell_id'])['QD'].median().median()
    lower = nominal * 0.80
    upper_qd = nominal * 1.20
    print(f"공칭 용량(nominal) : {nominal:.4f} Ah")
    print(f"필터 범위          : {lower:.4f} ~ {upper_qd:.4f} Ah")

    # 필터링
    df_clean = df[df['QD'].between(lower, upper_qd)].copy()
    for batch_id in batch_ids:
        n_all = (df['batch_id'] == batch_id).sum()
        n_clean = (df_clean['batch_id'] == batch_id).sum()
        print(f'Batch {batch_id}: 제거 {n_all - n_clean:,}행 ({(n_all-n_clean)/n_all*100:.2f}%)')

    # 시각화
    fig, axes = plt.subplots(1, len(batch_ids), figsize=(7 * len(batch_ids), 6),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    for ax, batch_id in zip(axes.flat, batch_ids):
        sub_batch = df_clean[df_clean['batch_id'] == batch_id]
        for _, sub in sub_batch.groupby('cell_id', sort=False):
            sub = sub.sort_values('cycle')
            life_value = sub['cycle_life'].iloc[0]
            color = cmap(norm(life_value)) if np.isfinite(life_value) else 'gray'
            ax.plot(sub['cycle'], sub['QD'], color=color, linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('QD (Ah)')
        ax.set_title(f'Batch {batch_id} - QD Curve (Filtered)')
        ax.axhline(nominal * 0.8, color='red', linestyle='--', label=f'EOL({nominal*0.8:.2f} Ah)')
        ax.set_ylim(0.7, 1.2)
        ax.legend()
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label='Cycle Life')
    plt.show()

if __name__ == '__main__':
    common.run_cli(main)

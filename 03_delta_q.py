from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

common = import_module('00_data')
display = common.display

# 수명별 1개 셀 - 특정 전압에서 Q(V) 차이

def read_delta_q(mat_file, cell_id):
    batch = mat_file['batch']
    summary = mat_file[batch['summary'][()].reshape(-1)[cell_id]]
    cycle_numbers = summary['cycle'][()].reshape(-1)
    index10 = np.flatnonzero(cycle_numbers == 10)
    index100 = np.flatnonzero(cycle_numbers == 100)
    if not len(index10) or not len(index100):
        return None
    cycles = mat_file[batch['cycles'][()].reshape(-1)[cell_id]]
    qd_refs = cycles['Qdlin'][()].reshape(-1)
    i10, i100 = int(index10[0]), int(index100[0])
    if max(i10, i100) >= len(qd_refs):
        return None
    q10 = mat_file[qd_refs[i10]][()].reshape(-1)
    q100 = mat_file[qd_refs[i100]][()].reshape(-1)
    voltage = mat_file[batch['Vdlin'][()].reshape(-1)[cell_id]][()].reshape(-1)
    if not (len(voltage) == len(q10) == len(q100)):
        raise ValueError(f'셀 {cell_id}: 전압과 용량 길이가 다릅니다.')
    valid = np.isfinite(voltage) & np.isfinite(q10) & np.isfinite(q100)
    voltage, q10, q100 = voltage[valid], q10[valid], q100[valid]
    if len(voltage) < 3:
        return None
    order = np.argsort(voltage)
    return voltage[order], q10[order], q100[order]


def main(context=None):
    context = context or common.get_context()
    df = context.df
    DATA_DIR = context.data_dir
    batch_ids, batch_names = context.batch_ids, context.batch_names
    cycle_life_df, life = context.cycle_life_df, context.life
    norm, cmap = context.norm, context.cmap

    sample_pool = cycle_life_df.dropna(subset=['cycle_life']).copy()
    sample_pool['수명 구분'] = pd.cut(sample_pool['cycle_life'],
        bins=[-np.inf, 500, 1000, np.inf], labels=['단수명', '중간수명', '장수명'])
    sample_records = []
    for batch_id in batch_ids:
        with h5py.File(DATA_DIR / batch_names[batch_id], 'r') as mat_file:
            for group in ['단수명', '중간수명', '장수명']:
                candidates = sample_pool[(sample_pool['batch_id'] == batch_id) &
                                         (sample_pool['수명 구분'] == group)].sample(frac=1, random_state=42)
                selected = 0
                for _, cell in candidates.iterrows():
                    values = read_delta_q(mat_file, int(cell['cell_id']))
                    if values is None:
                        continue
                    voltage, q10, q100 = values
                    indices = np.round(np.array([0.25, 0.50, 0.75]) * (len(voltage) - 1)).astype(int)
                    row = {'batch_id': batch_id, '수명 구분': group,
                           'cell_id': cell['cell_id'], 'cycle_life': cell['cycle_life']}
                    for i, index in enumerate(indices, start=1):
                        row[f'전압{i} (V)'] = voltage[index]
                        row[f'ΔQ{i} (Ah)'] = q100[index] - q10[index]
                    sample_records.append(row)
                    selected += 1
                    if selected == 1:
                        break
                if selected < 1:
                    print(f'Batch {batch_id} / {group}: 계산 가능한 셀 {selected}개')
    sample_delta_q_df = pd.DataFrame(sample_records)
    for batch_id in batch_ids:
        print(f'Batch {batch_id} - ΔQ = Q100 - Q10')
        if not sample_delta_q_df.empty:
            display(sample_delta_q_df[sample_delta_q_df['batch_id'] == batch_id].round(5))

    # 사이클 100번 - 사이클 10번의 Q(V) 차이
    fig, axes = plt.subplots(1, len(batch_ids), figsize=(7 * len(batch_ids), 6),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    delta_q_records = []
    for ax, batch_id in zip(axes.flat, batch_ids):
        sub = cycle_life_df[cycle_life_df['batch_id'] == batch_id]
        skipped = 0
        with h5py.File(DATA_DIR / batch_names[batch_id], 'r') as mat_file:
            for _, cell in sub.iterrows():
                values = read_delta_q(mat_file, int(cell['cell_id']))
                if values is None:
                    skipped += 1
                    continue
                voltage, q10, q100 = values
                delta_q = q100 - q10
                life_value = cell['cycle_life']
                color = cmap(norm(life_value)) if np.isfinite(life_value) else 'gray'
                ax.plot(voltage, delta_q, color=color, linewidth=0.8, alpha=0.7)
                delta_q_records.append({'batch_id': batch_id, 'source_file': cell['source_file'],
                    'cell_id': cell['cell_id'], 'cycle_life': life_value, 'V': voltage, 'delta_Q': delta_q})
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'Batch {batch_id} - ΔQ(V): Cycle 100 - Cycle 10')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('ΔQ (Ah)')
        print(f'Batch {batch_id}: 표시 {len(sub) - skipped}개 / 제외 {skipped}개')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label='Cycle Life')
    plt.show()

if __name__ == '__main__':
    common.run_cli(main)

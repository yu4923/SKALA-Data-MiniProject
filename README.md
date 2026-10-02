# ESS 배터리 수명 예측

초기 100사이클의 측정 데이터만으로 배터리의 최종 전체 수명인 cycle_life를 예측한다. EDA에서 확인한 용량 변화, 내부 저항, 온도와 충전 조건을 피쳐로 정리하고, 여러 회귀 모델의 배치 간 예측 성능을 비교하는 것이 목적이다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터: Batch 1 (2017-05-12)
- 평가 데이터: Batch 2 (2018-02-20)
- 추가 데이터: Batch 3 (2018-04-12), 현재는 EDA에 사용
- 태스크: Regression, cycle_life 예측
- 예측 시점: 100번째 사이클 종료 후
- 입력 범위: 1~100사이클. 이후의 측정값은 입력에 사용하지 않음
- 평가 기준: MAPE(%), 보조 지표 MAE·RMSE·R²

배터리별 한 행으로 정리한다. 100번째 사이클을 관측한 셀만 사용하며, 수명이 없는 셀은 학습·평가에서 제외한다.

| Batch | 전체 셀 수 | 수명이 있는 셀 수 | 모델링 용도 |
| --- | ---: | ---: | --- |
| 1 | 46 | 46 | 학습·Hold-out |
| 2 | 47 | 39 | 최종 평가 |
| 3 | 46 | 44 | EDA |

## 파일 구조

```text
python/
├── data/                       # MAT 데이터 3개, Git 제외
├── day1/
│   ├── 00_data.py
│   ├── 01_cycle_life.py
│   ├── 02_degradation.py
│   ├── 03_delta_q.py
│   ├── 04_charging.py
│   └── 05_correlation.py
├── day2/
│   ├── features.py
│   ├── training.py
│   ├── train_linear_regression.py
│   ├── train_random_forest.py
│   ├── train_catboost.py
│   ├── train_ridge.py
│   ├── train_svr.py
│   ├── train_extra_trees.py
│   ├── train_gradient_boosting.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   └── evaluate.py
├── models/                     # 전처리·모델·설정, Git 제외
│   └── baseline/               # 기존 기본 모델 보존
├── results/                    # 평가 지표·CV 지표·그래프, Git 제외
├── requirements.txt
└── README.md
```

학습 파일은 모델을 .pkl로 저장하고, evaluate.py는 저장된 모델들을 불러와 평가한다.

## 환경 설정

python 폴더에서 실행한다. 검증에는 Python 3.12를 사용했다.

```bash
python -m pip install -r requirements.txt
```

모델별 학습:

```bash
python day2/train_linear_regression.py
python day2/train_random_forest.py
python day2/train_catboost.py
python day2/train_ridge.py
python day2/train_svr.py
python day2/train_extra_trees.py
python day2/train_gradient_boosting.py
python day2/train_xgboost.py
python day2/train_lightgbm.py
```

저장된 모델의 Batch 2 평가:

```bash
python day2/evaluate.py
```

EDA 실행 예시:

```bash
python day1/01_cycle_life.py
```

데이터 기본 경로는 python/data이다. 다른 데이터 경로는 --data-dir으로 지정할 수 있다.

## EDA

### Cycle Life 분포

Batch별 히스토그램과 박스플롯, 장단수명 비율을 비교했다. 수명이 있는 셀 기준 분포는 다음과 같다.

| Batch | 최소 수명 | 중앙값 | 최대 수명 | 단수명 (<500) | 중간수명 (500~1000) | 장수명 (>1000) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 534 | 858.5 | 1,227 | 0개 (0.00%) | 36개 (78.26%) | 10개 (21.74%) |
| 2 | 392 | 472.0 | 1,186 | 28개 (71.79%) | 8개 (20.51%) | 3개 (7.69%) |
| 3 | 541 | 1,005.5 | 1,935 | 0개 (0.00%) | 21개 (47.73%) | 23개 (52.27%) |

핵심 발견: Batch 1에는 단수명 셀이 없고, Batch 2에는 28개가 있어 수명 분포가 크게 달랐다. Batch 3는 장수명 셀의 비율이 가장 높았다. 박스플롯의 장수명 이상치도 다른 셀과 초기 피쳐 평균을 비교했다.

### 열화 곡선 분석

전체 구간의 QD 곡선과 용량 이상값을 필터링한 곡선을 비교했다. 대부분 약 1.1Ah에서 시작하고, 장수명 셀은 높은 용량을 더 오래 유지했다. 곡선은 후반에 감소 기울기가 커지는 모습을 보였다.

필터링한 QD에 이동 중앙값을 적용하고, 각 셀의 초기 1/3과 마지막 1/3의 감소 기울기를 비교했을 때 Batch 1은 46/46개, Batch 2는 46/47개, Batch 3는 46/46개에서 후반 감소가 더 빨랐다. Knee point는 급격한 꺾임을 관찰하는 대상으로 보았으며, 셀별 발생 사이클을 수치로 추출하지는 않았다.

핵심 발견: 열화 속도는 일정하게 유지되기보다 수명 후반에 가속되는 경향이 나타났다. 이 전체 구간 분석은 EDA에만 사용하고, 예측 피쳐는 초기 100사이클로 제한했다.

### ΔQ(V) 곡선 분석

ΔQ(V) = Q100(V) − Q10(V)를 계산했다. 데이터에 저장된 Vdlin과 Qdlin의 같은 전압 지점을 사용하며, 임의의 전압으로 추가 보간하지 않았다. 수명별 대표 셀의 세 전압 지점과 전체 곡선을 비교했다.

수명이 짧은 셀에서 음수 방향의 변화가 큰 경향이 나타났다. Batch 3에서는 장수명 셀도 양수 방향으로 크게 변해 분산만으로 방향을 구분하기 어려웠다.

핵심 발견: 변화 폭을 나타내는 delta_Q_var와 음수 방향의 변화 정도를 나타내는 delta_Q_min을 함께 사용했다.

### 충전 속도(C-rate)와 수명의 관계

Batch별 충전 프로토콜의 평균 수명을 비교하고, 충전 전류와 수명의 관계를 확인했다. Batch 2에서는 newstructure가 붙은 일부 프로토콜의 평균 수명이 더 길게 나타났다.

핵심 발견: 프로토콜과 전류 패턴에 따라 수명이 달랐으며, 관계도 Batch별로 다르게 나타났다. 프로토콜은 범주형 피쳐로, 전류는 시간 가중 평균·표준편차·95백분위 값으로 정리했다. 전체 사이클 전류와 열화 속도의 비교는 EDA에만 사용했다.

### 내부 저항과 상관관계

단수명 셀의 IR 측정값은 0.017~0.018Ω 구간에 36.58%, 그 외 셀은 0.015~0.016Ω 구간에 39.74%가 분포했다. 0.019Ω 이상은 단수명 그룹 11.51%, 그 외 그룹 1.21%로 약 9.5배 차이가 났다. 이 비율은 배터리 수가 아닌 각 그룹의 측정값 수 기준이다.

Batch별 수명과의 Pearson 상관계수 절댓값이 가장 큰 피쳐는 Batch 1의 std_current_A(-0.901), Batch 2의 delta_Q_min(0.819), Batch 3의 p95_current_A(-0.708)이었다. 온도 피쳐 간에도 높은 상관이 나타났다.

핵심 발견: 수명과 강하게 연관된 피쳐가 Batch별로 달랐고, 서로 비슷한 정보를 가진 피쳐도 있었다.

## Modeling

### 피처 엔지니어링 전략

| 피쳐 | 계산 범위 | 사용 근거 |
| --- | --- | --- |
| mean_QD, std_QD, mean_QC | 1~100사이클 | 초기 용량 수준과 변화 폭 |
| mean_IR | 1~100사이클 | 단수명 그룹의 높은 저항 분포 |
| mean_Tmax, mean_Tavg, mean_Tmin | 1~100사이클 | 온도 특성과 피쳐 간 상관관계 비교 |
| mean_chargetime | 1~100사이클 | 충전 시간과 수명의 관계 |
| delta_Q_var, delta_Q_min | 100번 − 10번 사이클 | 변화 폭과 음수 방향의 변화를 함께 표현 |
| mean_current_A, std_current_A, p95_current_A | 10·50·100사이클 | 초기 충전 전류 패턴 |
| charging_policy | 충전 프로토콜 | 프로토콜별 수명 차이 |

수치형 피쳐의 결측값은 학습 데이터의 중앙값으로 채운다. CatBoost 이외의 모델에는 수치형 피쳐 표준화와 프로토콜 One-hot Encoding을 적용한다. CatBoost는 충전 프로토콜을 범주형으로 직접 처리한다. cell_id, batch_id, source_file은 입력 피쳐에서 제외한다.

### 모델 선택 및 근거

튜닝 설정은 Batch 1의 프로토콜별 5-fold CV로 선택했다. 아래 결과는 Batch 2 MAPE 순이며, 하이퍼파라미터가 다른 설정을 별도로 비교했다.

| 모델 / 설정 | 비교한 이유와 결과 |
| --- | --- |
| Gradient Boosting — 300회 | 얕은 트리로 예측 오차를 반복해서 보완하며, 지금까지 Batch 2 오차가 가장 낮았다. |
| LightGBM — 200회 | 리프 중심의 부스팅을 비교했으며, 300회보다 적은 반복으로 거의 같은 성능을 보였다. |
| LightGBM — 300회 | 반복 횟수를 늘렸을 때의 효과를 비교했으며, 200회와 오차 차이가 매우 작았다. |
| Gradient Boosting + LightGBM 평균 | 서로 다른 부스팅 모델의 예측이 보완되는지 비교했으며, 테스트한 앙상블 중 Batch 2 오차가 가장 낮았다. |
| Gradient Boosting — 200회 | 반복 횟수를 줄여 비교했으며, 300회보다 CV와 Hold-out 오차는 낮았지만 Batch 2 오차는 높았다. |

### 하이퍼파라미터 튜닝

단독 모델 9개에서 파라미터를 하나씩 바꾸며 총 135개 설정을 비교했다. 트리 수는 50~1,200개, 부스팅 반복 수는 최대 1,600회까지 확인했다. SVR은 최적화 반복 제한도 비교하고, 수렴하지 않은 설정은 선택에서 제외했다. 이후 평균 앙상블과 스태킹 9개 조합을 비교했다. 스태킹은 내부 CV에서도 충전 프로토콜을 분리하고, 교차검증 예측값을 Ridge로 다시 학습했다.

CatBoost는 별도로 기존 설정을 포함한 61개 조합을 비교했다. 반복 횟수 200~1,500회, 깊이 2~8, 학습률 0.01~0.1, 규제 강도 1~30 범위에서 조합을 추출했다. CV MAPE는 8.39%에서 7.67%로 감소했고, Batch 2 MAPE는 51.58%에서 57.13%로 증가했다.

현재 전체 탐색에서 CV 오차가 가장 낮은 설정은 SVR이며, Batch 2 오차가 가장 낮은 설정은 Gradient Boosting 300회이다. 반복 횟수를 늘리거나 여러 모델을 합친다고 오차가 계속 줄어들지는 않았다.

## 성능 결과

MAPE는 실제 수명 대비 절대 백분율 오차의 평균으로, 낮을수록 좋다. Train은 Hold-out을 제외한 35개 셀의 CV 평균, Valid는 별도 11개 셀, Test는 Batch 2의 39개 셀이다. 아래는 Batch 2 MAPE가 낮은 상위 5개 설정이다.

| 순위 | 모델 / 설정 | Train CV MAPE (%) | Valid MAPE (%) | Test MAPE (%) | 논문 대비 차이 (%p) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Gradient Boosting — 300회 | 8.44 | 6.61 | 32.67 | +23.57 |
| 2 | LightGBM — 200회 | 9.48 | 9.79 | 32.97 | +23.87 |
| 3 | LightGBM — 300회 | 9.48 | 9.79 | 32.97 | +23.87 |
| 4 | Gradient Boosting + LightGBM 평균 | 8.52 | 7.60 | 33.71 | +24.61 |
| 5 | Gradient Boosting — 200회 | 8.42 | 6.49 | 34.45 | +25.35 |

Gap은 Valid − Train, Test − Valid, Test − Target으로 계산하며, 단위는 %p이다. Target은 원논문 기준 9.1%를 사용했다.

논문의 데이터 정리와 train/test 분리는 이번 Batch 1 → Batch 2 구성과 다르므로, Gap은 논문 수치와의 비교이다. [원논문](https://www.nature.com/articles/s41560-019-0356-8), [저자 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)

### Gradient Boosting — 300회

설정: learning_rate=0.1, max_depth=1, min_samples_leaf=1, n_estimators=300

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 8.44 | 프로토콜별 5-fold CV 평균 |
| Valid (Batch 1 Hold-out) | 6.61 | 학습에 사용하지 않은 셀·프로토콜 |
| Test (Batch 2) | 32.67 | Batch 2 평가 |
| Gap (Train-Valid) | -1.83 | Valid − Train |
| Gap (Valid-Test) | +26.05 | Test − Valid |
| Gap (Target-Test) | +23.57 | Target: 원논문 9.1% |

### LightGBM — 200회

설정: learning_rate=0.1, num_leaves=7, min_child_samples=2, max_depth=-1, n_estimators=200

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 9.48 | 프로토콜별 5-fold CV 평균 |
| Valid (Batch 1 Hold-out) | 9.79 | 학습에 사용하지 않은 셀·프로토콜 |
| Test (Batch 2) | 32.97 | Batch 2 평가 |
| Gap (Train-Valid) | +0.32 | Valid − Train |
| Gap (Valid-Test) | +23.18 | Test − Valid |
| Gap (Target-Test) | +23.87 | Target: 원논문 9.1% |

### LightGBM — 300회

설정: learning_rate=0.1, num_leaves=7, min_child_samples=2, max_depth=-1, n_estimators=300

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 9.48 | 프로토콜별 5-fold CV 평균 |
| Valid (Batch 1 Hold-out) | 9.79 | 학습에 사용하지 않은 셀·프로토콜 |
| Test (Batch 2) | 32.97 | Batch 2 평가 |
| Gap (Train-Valid) | +0.32 | Valid − Train |
| Gap (Valid-Test) | +23.18 | Test − Valid |
| Gap (Target-Test) | +23.87 | Target: 원논문 9.1% |

### Gradient Boosting + LightGBM 평균

설정: Gradient Boosting 200회 + LightGBM 200회, 예측값을 동일 비중으로 평균

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 8.52 | 프로토콜별 5-fold CV 평균 |
| Valid (Batch 1 Hold-out) | 7.60 | 학습에 사용하지 않은 셀·프로토콜 |
| Test (Batch 2) | 33.71 | Batch 2 평가 |
| Gap (Train-Valid) | -0.92 | Valid − Train |
| Gap (Valid-Test) | +26.11 | Test − Valid |
| Gap (Target-Test) | +24.61 | Target: 원논문 9.1% |

### Gradient Boosting — 200회

설정: learning_rate=0.1, max_depth=1, min_samples_leaf=1, n_estimators=200

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 8.42 | 프로토콜별 5-fold CV 평균 |
| Valid (Batch 1 Hold-out) | 6.49 | 학습에 사용하지 않은 셀·프로토콜 |
| Test (Batch 2) | 34.45 | Batch 2 평가 |
| Gap (Train-Valid) | -1.93 | Valid − Train |
| Gap (Valid-Test) | +27.97 | Test − Valid |
| Gap (Target-Test) | +25.35 | Target: 원논문 9.1% |

Gradient Boosting 300회의 Test MAPE는 32.67%로 가장 낮았으며, 논문 목표보다 23.57%p 높았다. 평균 앙상블은 33.71%로 단독 모델의 최저 오차를 개선하지 못했다. 상위 5개 설정 모두 Batch 1보다 Batch 2에서 오차가 크게 증가했다.

## 오류 분석

### 큰 오차가 나타난 셀

앞서 평가한 Ridge는 Batch 2의 39개 중 29개 셀에서 음수 수명을 예측했다. 절대 오차가 큰 5개 셀은 모두 newstructure 프로토콜이었으며, 그중 2개는 초기 mean_IR이 0이었다.

테스트 오차가 가장 낮았던 Gradient Boosting의 절대 오차 상위 셀은 다음과 같다. cell_id는 원본 데이터의 0부터 시작하는 인덱스이다.

| Batch 2 cell_id | 실제 수명 | 예측 수명 | 예측 − 실제 | 절대 백분율 오차 (%) |
| --- | ---: | ---: | ---: | ---: |
| 34 | 1186 | 823.41 | -362.59 | 30.57 |
| 14 | 426 | 768.13 | +342.13 | 80.31 |
| 25 | 493 | 809.05 | +316.05 | 64.11 |
| 20 | 489 | 796.58 | +307.58 | 62.90 |
| 19 | 392 | 668.56 | +276.56 | 70.55 |

상위 5개 중 4개는 단수명 셀이었고, 수명을 실제보다 길게 예측했다. Gradient Boosting의 단수명 그룹 MAPE는 39.03%, 그 외 그룹은 16.47%였다.

### 데이터 분포와 오차

- 학습 데이터의 수명 범위는 534~1,227사이클이며, Batch 2의 30개 셀은 이 범위보다 짧았다.
- Batch 2의 32개 셀은 초기 mean_QD가 학습 범위를 벗어났다.
- Ridge의 CV MAPE는 7.64%, Hold-out MAPE는 26.66%, Batch 2 MAPE는 143.33%였다.

## ESS 도메인 해석

### BESS 운영 관점

이 모델은 초기 100사이클 데이터로 셀의 최종 전체 수명을 예측한다. 실험에서 Gradient Boosting의 절대 오차 상위 5개 셀 중 4개가 단수명 셀이었으며, 이들의 수명을 실제보다 길게 예측했다.

### 한계

분석에는 고속 충전 조건의 LFP/흑연 셀 실험 데이터를 사용했다. 실제 BESS 운영 데이터에 대한 학습·평가는 진행하지 않았다. Batch 2 최저 MAPE는 32.67%로, Batch 1 Hold-out의 6.61%보다 높았다. 기본 실습의 모델 성능 평가는 Batch 2까지 진행했으며, Batch 3는 EDA에 사용했다.


---

## 별도 실험: 단수명 셀을 추가한 혼합 학습

Batch 2의 단수명 셀 28개를 랜덤 시드 42로 14개씩 나누었다. 한쪽은 Batch 1에 추가해 학습하고, 나머지는 Batch 3에 추가해 테스트했다. 학습과 테스트의 셀은 중복되지 않는다. Batch 3에는 500사이클 미만 셀이 없다.

| 구분 | 데이터 구성 | 셀 수 |
| --- | --- | ---: |
| Train | Batch 1 46개 + Batch 2 단수명 14개 | 60 |
| Test | Batch 3 44개 + Batch 2의 나머지 단수명 14개 | 58 |

초기 100사이클 피쳐를 사용하고, 학습 데이터에서 구한 기준 용량의 80~120% 범위를 벗어난 QD를 제외한 뒤 mean_QD와 std_QD를 다시 계산했다. 모델별 하이퍼파라미터는 기존 설정을 유지했다.

### Random Forest 결과

9개 모델 중 전체 Test MAPE가 가장 낮았던 모델은 Random Forest였다.

| 평가 대상 | MAPE (%) |
| --- | ---: |
| 전체 테스트 58개 | 14.97 |
| Batch 3 44개 | 16.15 |
| Batch 2 단수명 14개 | 11.24 |

전체 테스트의 MAE는 163.76사이클이었다. MAPE 14.97%는 실제 수명 대비 절대 백분율 오차의 평균이 약 15%라는 뜻이다.

### 기존 학습 및 논문과 비교

| 비교 대상 | Test MAPE (%) | 혼합 학습 결과와의 차이 (%p) |
| --- | ---: | ---: |
| 논문 목표 | 9.10 | 혼합 학습이 5.87 높음 |
| 혼합 학습 Random Forest | 14.97 | — |
| Batch 1만 학습한 Random Forest, 동일한 테스트 58개 | 25.65 | 혼합 학습이 10.68 낮음 |
| 기존 QD 처리 Gradient Boosting, Batch 2의 39개 평가 | 31.31 | 혼합 학습이 16.34 낮음 |

동일한 테스트 데이터에서 Random Forest의 MAPE는 25.65%에서 14.97%로 약 41.6% 감소했고, MAE는 214.11사이클에서 163.76사이클로 줄었다. Batch 2 단수명 그룹의 MAPE는 56.30%에서 11.24%로 감소했고, Batch 3의 MAPE는 15.89%에서 16.15%로 증가했다.

이번 실험은 학습과 테스트에 서로 다른 Batch 2 셀이 포함된 혼합 구성이다. 기존 Batch 1 → Batch 2 평가 및 논문과는 데이터 분리가 다르므로, 위 수치 차이는 동일한 실험 조건의 성능 차이가 아니다.

실행 파일: experiments/mixed_batches.py

```bash
python experiments/mixed_batches.py
```

## Contributor

김태완

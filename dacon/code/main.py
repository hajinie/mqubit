import option as op
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'TEST_SIZE': 0.2,  # 테스트 사이즈
    'N_ESTIMATORS': 100,  # RandomForest의 트리 개수
}

# Seed 고정
op.seed_everything(CFG['SEED'])

# 학습 데이터 로드 및 전처리
chembl_data = pd.read_csv('../data/train.csv')
chembl_data['Fingerprint'] = chembl_data['Smiles'].apply(op.smiles_to_fingerprint)
train_x = np.stack(chembl_data['Fingerprint'].values)
train_y = chembl_data['pIC50'].values

# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=CFG['TEST_SIZE'], random_state=CFG['SEED'])

# 랜덤 포레스트 모델 정의 및 학습
model = RandomForestRegressor(
    n_estimators=CFG['N_ESTIMATORS'],
    random_state=CFG['SEED']
)
model.fit(train_x, train_y)

# 검증 데이터에 대한 예측 및 평가
val_y_pred = model.predict(val_x)
rmse = op.rmse(val_y,val_y_pred)
r2_val = r2_score(val_y, val_y_pred)
final_score = op.score(val_y, val_y_pred)

print(f'RMSE: {rmse:.4f}')
print(f'R2_val: {r2_val:.4f}')
print(f"Final Score: {final_score:.4f}")

# 테스트 데이터 예측 및 제출 파일 생성
test_data = pd.read_csv('../data/test.csv')
test_data['Fingerprint'] = test_data['Smiles'].apply(op.smiles_to_fingerprint)
test_x = np.stack(test_data['Fingerprint'].values)
test_y_pred = model.predict(test_x)

submit = pd.read_csv('../data/sample_submission.csv')
submit['IC50_nM'] = op.pIC50_to_IC50(test_y_pred)
submit.to_csv('../data/SUBMIT.csv', index=False)

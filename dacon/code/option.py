
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error


CFG = {
    'NBITS': 2048,
    'SEED': 42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        return np.array(fp)
    else:
        return np.zeros(CFG['NBITS'])

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

def rmse(y_true, y_pred):
    mse = mean_squared_error(pIC50_to_IC50(y_true), pIC50_to_IC50(y_pred))
    rmse = np.sqrt(mse)
    return rmse

def normalized_rmse(y_true, y_pred):
    mse = mean_squared_error(pIC50_to_IC50(y_true), pIC50_to_IC50(y_pred))
    rmse = np.sqrt(mse)
    normalized_rmse = rmse / (np.max(y_true) - np.min(y_true))
    return normalized_rmse

def correct_ratio(y_true, y_pred):
    absolute_error = np.abs(y_true - y_pred)
    correct_ratio = np.mean(absolute_error <= 0.5)
    return correct_ratio

def score(y_true, y_pred):
    A = normalized_rmse(y_true, y_pred)
    B = correct_ratio(y_true, y_pred)
    
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return score

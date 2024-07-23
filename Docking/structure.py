from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# SMILES 파일을 읽어들이는 함수
def read_smi_file(file_path):
    with open(file_path, 'r') as file:
        smiles_list = file.readlines()
    return [smiles.strip() for smiles in smiles_list]

# SMILES 문자열을 RDKit 분자로 변환
def smiles_to_molecules(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# 분자 구조를 시각화
def visualize_molecules(molecules):
    img = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(200,200))
    img.save("molecules_grid.png")  # 이미지 파일로 저장
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# SMILES 파일 경로
file_path = './Bitter.smi'

# 파일을 읽고 분자로 변환 및 시각화
smiles_list = read_smi_file(file_path)
molecules = smiles_to_molecules(smiles_list)
visualize_molecules(molecules)


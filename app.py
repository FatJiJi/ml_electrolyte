import streamlit as st
import torch
import pandas as pd
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ------------------------
# 路径设置
MODEL_PATH = "conductivity_fnn.pt"
CONFIG_PATH = "model_meta.json"

# ------------------------
# 加载模型和配置
@st.cache_resource  # Streamlit 会缓存模型，避免每次刷新都加载
def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache_data
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config

model = load_model()
config = load_config()

# ------------------------
# Morgan 指纹生成
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ------------------------
# 特征构建
def build_features(salt_smiles, solvent_smiles_list, ratios):
    # salt fingerprint
    salt_fp = smiles_to_morgan(salt_smiles)
    if salt_fp is None:
        return None

    # solvent fingerprints
    solvent_fps = []
    for s in solvent_smiles_list:
        if s.strip() == "":
            continue
        fp = smiles_to_morgan(s)
        if fp is None:
            return None
        solvent_fps.append(fp)
    
    # 按比例加权平均
    if len(solvent_fps) > 0:
        ratios = np.array(ratios[:len(solvent_fps)])
        ratios = ratios / ratios.sum()  # 归一化
        solvent_fp = np.sum([fp*rat for fp,rat in zip(solvent_fps, ratios)], axis=0)
    else:
        solvent_fp = np.zeros_like(salt_fp)
    
    # 合并 salt + solvent
    features = np.concatenate([salt_fp, solvent_fp])
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return features

# ------------------------
# Streamlit 前端
st.title("Electrolyte Conductivity Predictor")

st.markdown("输入电解液配方，预测电导率 (ms/cm)")

salt_smiles = st.text_input("盐 SMILES", value="[Li+].F[P-](F)(F)(F)(F)F")
solvent1 = st.text_input("溶剂1 SMILES", value="C1COC(=O)O1")
ratio1 = st.number_input("溶剂1比例", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
solvent2 = st.text_input("溶剂2 SMILES", value="COC(=O)OC")
ratio2 = st.number_input("溶剂2比例", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
solvent3 = st.text_input("溶剂3 SMILES", value="")
ratio3 = st.number_input("溶剂3比例", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

ratios = [ratio1, ratio2, ratio3]
solvents = [solvent1, solvent2, solvent3]

if st.button("预测电导率"):
    features = build_features(salt_smiles, solvents, ratios)
    if features is None:
        st.error("SMILES 无法解析，请检查输入！")
    else:
        with torch.no_grad():
            y_pred = model(features)
        st.success(f"预测电导率: {y_pred.item():.4f} ms/cm")

import streamlit as st
import torch
import torch.nn as nn
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ------------------------
# 路径设置
MODEL_PATH = "conductivity_fnn.pt"
CONFIG_PATH = "model_meta.json"

# ------------------------
# 定义 FNN 模型结构（必须和训练时一致）
class FNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# 加载模型和配置
@st.cache_resource
def load_model():
    # 先读取配置文件获取输入维度
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    input_dim = config['input_dim']
    
    model = FNN(input_dim=input_dim)
    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

model, config = load_model()

# ------------------------
# Morgan 指纹生成
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ------------------------
# 构建特征（含浓度）
def build_features(salt_smiles, solvent_smiles_list, ratios, concentration, n_bits=1024):
    salt_fp = smiles_to_morgan(salt_smiles)
    if salt_fp is None:
        return None

    solvent_fps = []
    for s in solvent_smiles_list:
        if s.strip() == "":
            continue
        fp = smiles_to_morgan(s)
        if fp is None:
            return None
        solvent_fps.append(fp)

    if len(solvent_fps) > 0:
        ratios = np.array(ratios[:len(solvent_fps)])
        ratios = ratios / ratios.sum()
        solvent_fp = np.sum([fp * r for fp, r in zip(solvent_fps, ratios)], axis=0)
    else:
        solvent_fp = np.zeros_like(salt_fp)

    conc = np.array([concentration], dtype=np.float32)  # 使用用户输入的浓度
    features = np.concatenate([salt_fp, solvent_fp, conc])
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
concentration = st.number_input("盐浓度 (M)", min_value=0.0, value=1.0, step=0.1)

ratios = [ratio1, ratio2, ratio3]
solvents = [solvent1, solvent2, solvent3]

if st.button("预测电导率"):
    features = build_features(salt_smiles, solvents, ratios, concentration, n_bits=config["n_bits"])
    if features is None:
        st.error("SMILES 无法解析，请检查输入！")
    else:
        with torch.no_grad():
            y_pred = model(features)
            st.success(f"预测电导率: {y_pred.item():.4f} ms/cm")

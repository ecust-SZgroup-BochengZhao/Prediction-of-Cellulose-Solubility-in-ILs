import pickle
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ''

def bitvect_to_np(bv, n_bits):
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def compute_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.int8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return bitvect_to_np(bv, n_bits).astype(np.float32)

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CelluloseSolubilityPredictor:
    def __init__(self, artifact_path: str):
        with open(artifact_path, "rb") as f:
            art = pickle.load(f)

        self.meta = art["meta"]
        self.temp_scaler = art["temp_scaler"]
        self.folds = []

        for fold_pack in art["folds"]:
            p = fold_pack["params"]
            model = MLPModel(
                input_dim=self.meta["input_dim"],
                hidden_dim=p["hidden_dim"],
                output_dim=1,
                num_layers=p["num_layers"],
                dropout_rate=p["dropout_rate"]
            ).to(device)
            model.load_state_dict(fold_pack["state_dict"])
            model.eval()
            self.folds.append({"fold_id": fold_pack["fold_id"], "params": p, "model": model})

        self.radius = self.meta["radius"]
        self.n_bits = self.meta["n_bits"]

    def _featurize(self, cation_smiles: str, anion_smiles: str, temp_c: float) -> np.ndarray:
        cation_smiles = standardize_smiles(str(cation_smiles))
        anion_smiles  = standardize_smiles(str(anion_smiles))

        c_fp = compute_morgan_fingerprint(cation_smiles, self.radius, self.n_bits)
        a_fp = compute_morgan_fingerprint(anion_smiles,  self.radius, self.n_bits)
        fp = np.hstack([c_fp, a_fp]).astype(np.float32)

        t_scaled = self.temp_scaler.transform(np.array([[float(temp_c)]], dtype=np.float32)).astype(np.float32).ravel()
        x = np.hstack([fp, t_scaled]).astype(np.float32)
        return x

    def predict(self, cation_smiles: str, anion_smiles: str, temp_c: float,
                return_std: bool = False, return_all: bool = False):
        x = self._featurize(cation_smiles, anion_smiles, temp_c)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        preds = []
        with torch.no_grad():
            for fold in self.folds:
                y_hat = fold["model"](x_t).cpu().numpy().ravel()[0]
                preds.append(float(y_hat))

        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds))

        if return_all and return_std:
            return mean_pred, std_pred, preds
        if return_all:
            return mean_pred, preds
        if return_std:
            return mean_pred, std_pred
        return mean_pred


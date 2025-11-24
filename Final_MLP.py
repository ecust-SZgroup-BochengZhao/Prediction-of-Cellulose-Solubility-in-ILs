import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rdkit import Chem
from rdkit.Chem import AllChem


file_path = 'database.xlsx'
database = pd.read_excel(file_path)

def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return ''

database['Cation_SMILES'] = database['Cation_SMILES'].apply(standardize_smiles)
database['Anion_SMILES'] = database['Anion_SMILES'].apply(standardize_smiles)



def compute_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else np.zeros(n_bits)


database['Cation_Fingerprint'] = database['Cation_SMILES'].apply(lambda x: compute_morgan_fingerprint(x))
database['Anion_Fingerprint'] = database['Anion_SMILES'].apply(lambda x: compute_morgan_fingerprint(x))


cation_fp = np.array(database['Cation_Fingerprint'].tolist())
anion_fp = np.array(database['Anion_Fingerprint'].tolist())
fingerprints = np.hstack([cation_fp, anion_fp])

scaler = StandardScaler()
database['Temperature'] = scaler.fit_transform(database[['T/(℃)']])
features = np.hstack([fingerprints, database[['Temperature']].values])

X = features
y = database['S.mol.log'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(MLPModel, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


best_params = {
    'learning_rate': 0.0016873495104568939,
    'hidden_dim': 243,
    'num_layers': 3,
    'dropout_rate': 0.32802446803574603,
    'weight_decay': 2.447405744579684e-05
}

num_epochs = 200

model = MLPModel(
    input_dim=features.shape[1],
    hidden_dim=best_params['hidden_dim'],
    output_dim=1,
    num_layers=best_params['num_layers'],
    dropout_rate=best_params['dropout_rate']
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)
criterion = torch.nn.MSELoss()


X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor).flatten()
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()



model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).flatten().cpu().numpy()

y_true = y

final_mae = mean_absolute_error(y_true, y_pred)
final_r2 = r2_score(y_true, y_pred)

print(f"Final MAE (train on full data): {final_mae:.4f}")
print(f"Final R2  (train on full data): {final_r2:.4f}")
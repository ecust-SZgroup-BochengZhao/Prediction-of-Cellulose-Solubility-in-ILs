from Predictor import CelluloseSolubilityPredictor

# 1)  Load Model
predictor = CelluloseSolubilityPredictor("Final_MLPmodel.pkl")

# 2) predict: input cation SMILES, anion SMILE, temp(℃)
y_mean = predictor.predict(
    cation_smiles="CN1C=C[N+](=C1)C",
    anion_smiles="CC(=O)[O-]",
    temp_c=80.0
)

print("5-fold mean prediction:", y_mean)
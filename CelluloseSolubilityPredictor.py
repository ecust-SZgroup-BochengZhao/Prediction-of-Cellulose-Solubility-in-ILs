from Predictor import CelluloseSolubilityPredictor

def main():
    # 1) Load Model
    predictor = CelluloseSolubilityPredictor("Final_MLPmodel.pkl")

    # 2) Manual input
    cation_smiles = input("Enter cation SMILES: ").strip()
    anion_smiles  = input("Enter anion SMILES : ").strip()
    temp_str      = input("Enter temperature (°C): ").strip()

    if not cation_smiles or not anion_smiles:
        raise ValueError("Cation/anion SMILES cannot be empty.")

    try:
        temp_c = float(temp_str)
    except ValueError:
        raise ValueError(f"Temperature must be a number, got: {temp_str!r}")

    # 3) Predict
    y_mean = predictor.predict(
        cation_smiles=cation_smiles,
        anion_smiles=anion_smiles,
        temp_c=temp_c
    )

    print("5-fold mean prediction:", y_mean)

if __name__ == "__main__":
    main()

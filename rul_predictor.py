import joblib
import pandas as pd

class RULPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.features = self.model.feature_names_in_.tolist()

    def predict(self, data: dict) -> float:
        """
        Predict Remaining Useful Life (RUL) in hours
        """
        df = pd.DataFrame([data])

        # Ensure correct feature order
        df = df[self.features]

        prediction = self.model.predict(df)[0]
        return float(prediction)

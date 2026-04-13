import joblib
import pandas as pd
import os

def get_prediction(house_data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "house_price_model.pkl")

    pipeline = joblib.load(model_path)
    input_df = pd.DataFrame([house_data])
    prediction = pipeline.predict(input_df)[0]
    
    return prediction
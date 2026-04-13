from sklearn.model_selection import train_test_split
from src.preprocessing import get_preprocessor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd
import joblib
import os

def run_training():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_path = os.path.join(project_root, "data", "train.csv")
    model_dir = os.path.join(project_root, "models")
    model_save_path = os.path.join(model_dir, "house_price_model.pkl")

    df = pd.read_csv(data_path)
    df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)
    X = df.drop(["SalePrice"], axis=1)
    y = df["SalePrice"]

    categorical_cols_with_na = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual",
        "GarageCond", "PoolQC", "Fence", "MiscFeature"
    ]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    preprocessor = get_preprocessor(categorical_cols_with_na)

    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        n_jobs=-1,
        early_stopping_rounds=50,
        random_state=42,
        colsample_bytree=0.7,
        max_depth=5,
        subsample=0.8
    )
    
    X_train_prepped = preprocessor.fit_transform(X_train)
    X_valid_prepped = preprocessor.transform(X_valid)

    model.fit(
        X_train_prepped, y_train,
        eval_set=[(X_valid_prepped, y_valid)],
        verbose=False
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    joblib.dump(full_pipeline, model_save_path)
    print(f"Training complete. Model saved to: {model_save_path}")

if __name__ == "__main__":
    run_training()
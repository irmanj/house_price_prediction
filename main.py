import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    X = df[["luas", "kamar"]]
    y = df["harga"]
    return X, y

def train(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    return LinearRegression()

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE    : {mae:.2f}")
    print(f"RMSE    : {rmse:.2f}")

def save_model(model):
    joblib.dump(model, "model.pkl")

def main():
    df = load_data("data.csv")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
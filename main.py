import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

def load_data():
    return pd.read_csv("data.csv")

def train_model(df):
    X = df[["luas", "kamar"]]
    y = df["harga"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print("MAE:", mae)

    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    df = load_data()
    train_model(df)
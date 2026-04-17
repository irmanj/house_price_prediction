import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import pickle

# load data
df = pd.read_csv("data.csv")

# fitur dan target
X = df[["luas_rumah", "jumlah_kamar", "lokasi"]]
y = df["harga"]

# encoding lokasi
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), ["lokasi"])
    ],
    remainder="passthrough"
)

# pipeline
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved!")
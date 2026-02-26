import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Progression Prediction")

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

st.write(f"MSE: {mean_squared_error(y_test, pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, pred):.2f}")

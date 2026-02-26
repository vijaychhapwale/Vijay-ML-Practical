import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="KNN Weather Classification", layout="wide")
st.title("🌦️ K-Nearest Neighbor Weather Classification")

X = np.array([[50,70],[25,80],[27,60],[31,65],[23,85],[20,75]])
y = np.array([0,1,0,0,1,1])
labels = {0: "Sunny", 1: "Rainy"}

st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("Temperature (°C)", 10, 60, 26)
hum = st.sidebar.slider("Humidity (%)", 50, 95, 78)
k = st.sidebar.slider("K Value", 1, 5, 3)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)
pred = model.predict([[temp, hum]])[0]

st.sidebar.success(f"Prediction: {labels[pred]}")

fig, ax = plt.subplots()
ax.scatter(X[y==0,0], X[y==0,1], label="Sunny", s=100)
ax.scatter(X[y==1,0], X[y==1,1], label="Rainy", s=100)
ax.scatter(temp, hum, marker="*", s=300, label="New Day")
ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

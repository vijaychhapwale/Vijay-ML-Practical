import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="ID3 Decision Tree", layout="centered")
st.title("🌳 ID3 Decision Tree Classifier")

data = pd.DataFrame({
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain",
                "Overcast","Sunny","Sunny","Rain","Sunny","Overcast",
                "Overcast","Rain"],
    "Humidity": ["High","High","High","High","Normal","Normal",
                 "Normal","High","Normal","High","Normal","High",
                 "Normal","High"],
    "PlayTennis": ["No","No","Yes","Yes","No","Yes",
                    "Yes","No","Yes","Yes","Yes","Yes",
                    "Yes","No"]
})

st.subheader("📊 Training Dataset")
st.dataframe(data, use_container_width=True)

def entropy(col):
    _, counts = np.unique(col, return_counts=True)
    return -sum((c/len(col))*math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total = entropy(df[target])
    vals = df[attr].unique()
    return total - sum(
        (len(df[df[attr]==v])/len(df))*entropy(df[df[attr]==v][target])
        for v in vals
    )

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda x: info_gain(df, x, target))
    tree = {best: {}}
    for v in df[best].unique():
        tree[best][v] = id3(df[df[best]==v], target, [a for a in attrs if a!=best])
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    key = next(iter(tree))
    return predict(tree[key][sample[key]], sample)

if st.button("Generate Decision Tree"):
    tree = id3(data, "PlayTennis", ["Outlook","Humidity"])
    st.session_state.tree = tree
    st.subheader("🌲 Generated Decision Tree")
    st.json(tree)

if "tree" in st.session_state:
    st.subheader("🔮 Prediction")
    o = st.selectbox("Outlook", data["Outlook"].unique())
    h = st.selectbox("Humidity", data["Humidity"].unique())

    if st.button("Predict"):
        result = predict(st.session_state.tree, {"Outlook": o, "Humidity": h})

        result_df = pd.DataFrame({
            "Outlook": [o],
            "Humidity": [h],
            "PlayTennis": [result]
        })

        st.table(result_df)
        st.success(f"Prediction Result: {result}")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Titanic Survival", layout="centered")
st.title("🚢 Titanic Survival Prediction")

# File uploader (supports CSV and Excel)
file = st.file_uploader("Upload Titanic Dataset (CSV or XLSX)", type=["csv", "xlsx"])

if file is not None:
    try:
        # Read file based on extension
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            st.error("Unsupported file format. Please upload CSV or XLSX.")
            st.stop()

        st.subheader("Dataset Preview")
        st.write(df.head())

        # Data Preprocessing
        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(df["Age"].median())

        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())

        if "Embarked" in df.columns:
            df = df.dropna(subset=["Embarked"])

        # Convert categorical to numeric
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

        # Required features check
        required_features = [
            "Pclass", "Age", "SibSp", "Parch", "Fare",
            "Sex_male", "Embarked_Q", "Embarked_S"
        ]

        missing = [col for col in required_features if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        if "Survived" not in df.columns:
            st.error("Target column 'Survived' not found in dataset.")
            st.stop()

        # Features and target
        X = df[required_features]
        y = df["Survived"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model training
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Output
        st.success(f"Model Accuracy: {accuracy:.2f}")

    except Exception as e:
        st.error(f"Error reading file: {e}")

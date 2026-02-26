import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Spam Email Detection", layout="centered")
st.title("📧 Spam Email Detection")

emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Claim your prize immediately",
    "Project discussion with team",
    "Limited offer buy now"
]

labels = [1, 0, 1, 0, 1]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

model = LinearSVC()
model.fit(X, labels)

st.subheader("Enter Email Message")
message = st.text_area("Email Text")

if st.button("Check Spam"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)[0]
        st.success("Spam Email 🚫" if prediction == 1 else "Not Spam Email ✅")

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.write("Detect whether a news article is REAL or FAKE using Machine Learning.")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():

    fake = pd.read_csv("D:/Data Science/ML Model/New Detector/fake.csv")
    true = pd.read_csv("D:/Data Science/ML Model/New Detector/true.csv")

    # Labels
    fake["labels"] = 0
    true["labels"] = 1

    # Combine datasets
    data = pd.concat([fake, true])

    # Shuffle data
    data = data.sample(frac=1, random_state=42)

    return data

data = load_data()

# -----------------------------------
# PREPARE DATA
# -----------------------------------
x = data["text"]
y = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# TF-IDF VECTORIZER
# -----------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

x_train_prepared = vectorizer.fit_transform(x_train)
x_test_prepared = vectorizer.transform(x_test)

# -----------------------------------
# LOGISTIC REGRESSION MODEL
# -----------------------------------
log_model = LogisticRegression(max_iter=1000)

log_model.fit(x_train_prepared, y_train)

log_predictions = log_model.predict(x_test_prepared)

log_accuracy = accuracy_score(y_test, log_predictions) * 100

# -----------------------------------
# PASSIVE AGGRESSIVE MODEL
# -----------------------------------
passive_model = PassiveAggressiveClassifier(max_iter=1000)

passive_model.fit(x_train_prepared, y_train)

passive_predictions = passive_model.predict(x_test_prepared)

passive_accuracy = accuracy_score(y_test, passive_predictions) * 100

# -----------------------------------
# SHOW ACCURACY
# -----------------------------------
st.subheader("📊 Model Accuracy")

st.write(f"✅ Logistic Regression Accuracy: **{log_accuracy:.2f}%**")

st.write(f"✅ Passive Aggressive Accuracy: **{passive_accuracy:.2f}%**")

# -----------------------------------
# MODEL SELECTION
# -----------------------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Passive Aggressive", "Logistic Regression"]
)

if model_choice == "Passive Aggressive":
    selected_model = passive_model
else:
    selected_model = log_model

# -----------------------------------
# USER INPUT
# -----------------------------------
st.subheader("📝 Enter News Text")

user_input = st.text_area(
    "Paste News Article Here",
    height=250,
    placeholder="Enter news text..."
)

# -----------------------------------
# PREDICT BUTTON
# -----------------------------------
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some news text.")

    else:

        input_data = [user_input]

        vectorized_input = vectorizer.transform(input_data)

        prediction = selected_model.predict(vectorized_input)

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("🚨 This News is likely FAKE")
        else:
            st.success("✅ This News is likely REAL")

# -----------------------------------
# SAMPLE NEWS
# -----------------------------------
st.subheader("📰 Sample News")

sample_news = [
    "The Indian Space Research Organisation (ISRO) successfully launched a new weather monitoring satellite from Sriharikota on Friday.",

    "Scientists confirm that drinking melted ice cream every morning can increase human lifespan to 150 years.",

    "The Reserve Bank of India announced that inflation rates have remained within the target range for the third consecutive quarter."
]

for sample in sample_news:

    if st.button(sample[:70] + "..."):

        vec = vectorizer.transform([sample])

        pred = selected_model.predict(vec)

        st.info(sample)

        if pred[0] == 0:
            st.error("Prediction: FAKE")
        else:
            st.success("Prediction: REAL")
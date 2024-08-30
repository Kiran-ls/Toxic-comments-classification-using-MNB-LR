import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Load the datasets
@st.cache_data()
def load_data():
    train_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/toxic comments/train.csv')
    test_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/toxic comments/test.csv')
    return train_df.copy(), test_df.copy()


train_df, test_df = load_data()


# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ')
    return text


train_df['comment_text'] = train_df['comment_text'].map(lambda com: clean_text(com))
test_df['comment_text'] = test_df['comment_text'].map(lambda com: clean_text(com))

# Split the datasets
X_train, y_train = train_df['comment_text'], train_df.drop(columns=['id', 'comment_text'])
X_test, y_test = test_df['comment_text'], test_df.drop(columns=['id', 'comment_text'])
X_train, X_test, y_train, y_test = train_test_split(train_df['comment_text'],
                                                    train_df.drop(columns=['id', 'comment_text']), test_size=0.2,
                                                    random_state=42)

# Vectorize text data
vect = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Train Naive Bayes classifier for each label
nb_classifiers = {}
for label in y_train.columns:
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_dtm, y_train[label])
    nb_classifiers[label] = nb_classifier

# Train Logistic Regression classifier for each label
lr_classifiers = {}
for label in y_train.columns:
    lr_classifier = LogisticRegression(max_iter=200)
    lr_classifier.fit(X_train_dtm, y_train[label])
    lr_classifiers[label] = lr_classifier

# Hybrid Model for each label
hybrid_predictions_test = {}
for label in y_train.columns:
    nb_test_probs = nb_classifiers[label].predict_proba(X_test_dtm)[:, 1]
    lr_test_probs = lr_classifiers[label].predict_proba(X_test_dtm)[:, 1]
    test_probs = (nb_test_probs + lr_test_probs) / 2
    hybrid_pred_test = np.where(test_probs > 0.5, 1, 0)
    hybrid_predictions_test[label] = hybrid_pred_test

# Create Streamlit UI
st.title("Toxic Comments Classifier")

# Display user input text area
text = st.text_area("Enter a comment:", "")

# Display classifier options
option = st.selectbox("Select a model", ("Naive Bayes", "Logistic Regression", "Hybrid"))

# Make predictions and display results
if st.button("Classify"):
    input_text = clean_text(text)
    input_text_dtm = vect.transform([input_text])

    if option == "Naive Bayes":
        predictions = {}
        for label, classifier in nb_classifiers.items():
            prediction = classifier.predict(input_text_dtm)
            predictions[label] = prediction[0]
        st.subheader(" Naive Bayes Predictions:")
        for label, prediction in predictions.items():
            st.write(f"{label}: {prediction}")
    elif option == "Logistic Regression":
        predictions = {}
        for label, classifier in lr_classifiers.items():
            prediction = classifier.predict(input_text_dtm)
            predictions[label] = prediction[0]
        st.subheader(" Logistic Regression Predictions:")
        for label, prediction in predictions.items():
            st.write(f"{label}: {prediction}")
    else:  # Hybrid Model
        hybrid_predictions = {}
        for label in y_train.columns:
            nb_test_probs = nb_classifiers[label].predict_proba(input_text_dtm)[:, 1]
            lr_test_probs = lr_classifiers[label].predict_proba(input_text_dtm)[:, 1]
            test_probs = (nb_test_probs + lr_test_probs) / 2
            hybrid_pred = np.where(test_probs > 0.5, 1, 0)
            hybrid_predictions[label] = hybrid_pred
        st.write("Hybrid Model Prediction:")
        for label, prediction in hybrid_predictions.items():
                st.write(f"{label}: {prediction}")
        #st.subheader("Predictions:")
        #for label, prediction in predictions.items():
         #  st.write(f"{label}: {prediction}")

# Display distribution of toxic comments
st.subheader("Distribution of Toxic Comments")
toxic_counts = train_df.iloc[:, 2:].sum()
st.bar_chart(toxic_counts)





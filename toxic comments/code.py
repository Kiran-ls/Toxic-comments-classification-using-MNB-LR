import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re
import joblib
import matplotlib.pyplot as plt

# Load the datasets
train_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/toxic comments/train.csv')
test_df = pd.read_csv('C:/Users/admin/OneDrive/Desktop/toxic comments/test.csv')

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
print(train_df["comment_text"].isna().sum())
print(test_df["comment_text"].isna().sum())

# Split the datasets
X_train, y_train = train_df['comment_text'], train_df.drop(columns=['id', 'comment_text'])
X_test, y_test = test_df['comment_text'], test_df.drop(columns=['id', 'comment_text'])
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(train_df['comment_text'],
                                                    train_df.drop(columns=['id', 'comment_text']), test_size=0.2,
                                                    random_state=42)

# Print the column names of both training and testing datasets
print("Training dataset columns:", y_train.columns)
print("Testing dataset columns:", y_test.columns)

# Vectorize text data
vect = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Train Naive Bayes classifier for each label with Laplace smoothing
nb_classifiers = {}
nb_training_accuracies = []  # Store training accuracies for Naive Bayes
nb_test_accuracies = []  # Store test accuracies for Naive Bayes
nb_classification_reports = {}

for label in y_train.columns:
    nb_classifier = MultinomialNB(alpha=1.0)  # Laplace smoothing
    nb_classifier.fit(X_train_dtm, y_train[label])
    nb_classifiers[label] = nb_classifier

    y_train_pred_nb = nb_classifier.predict(X_train_dtm)
    y_test_pred_nb = nb_classifier.predict(X_test_dtm)
    nb_classification_reports[label] = classification_report(y_test[label], y_test_pred_nb)

    train_accuracy = accuracy_score(y_train[label], y_train_pred_nb)
    test_accuracy = accuracy_score(y_test[label], y_test_pred_nb)

    nb_training_accuracies.append(train_accuracy)
    nb_test_accuracies.append(test_accuracy)

    print(f"Naive Bayes Classifier for {label}:")
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Classification Report:")
    print(nb_classification_reports[label])

# Calculate overall Naive Bayes accuracy
overall_nb_training_accuracy = sum(nb_training_accuracies) / len(nb_training_accuracies)
overall_nb_test_accuracy = sum(nb_test_accuracies) / len(nb_test_accuracies)
print("Naive Bayes Training Accuracy:", overall_nb_training_accuracy)
print("Naive Bayes Test Accuracy:", overall_nb_test_accuracy)

# Train Logistic Regression classifier for each label with L2 regularization
lr_classifiers = {}
lr_training_accuracies = []  # Store training accuracies for Logistic Regression
lr_test_accuracies = []  # Store test accuracies for Logistic Regression
lr_classification_reports = {}

for label in y_train.columns:
    lr_classifier = LogisticRegression(max_iter=200, penalty='l2', solver='liblinear', C=1.0)
    lr_classifier.fit(X_train_dtm, y_train[label])
    lr_classifiers[label] = lr_classifier

    y_train_pred_lr = lr_classifier.predict(X_train_dtm)
    y_test_pred_lr = lr_classifier.predict(X_test_dtm)
    lr_classification_reports[label] = classification_report(y_test[label], y_test_pred_lr)

    train_accuracy = accuracy_score(y_train[label], y_train_pred_lr)
    test_accuracy = accuracy_score(y_test[label], y_test_pred_lr)

    lr_training_accuracies.append(train_accuracy)
    lr_test_accuracies.append(test_accuracy)

    print(f"\nLogistic Regression Classifier for {label}:")
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Classification Report:")
    print(lr_classification_reports[label])

# Calculate overall training and test accuracies for logistic regression
lr_overall_train_accuracy = sum(lr_training_accuracies) / len(lr_training_accuracies)
lr_overall_test_accuracy = sum(lr_test_accuracies) / len(lr_test_accuracies)
print("\n Logistic Regression Training Accuracy:", lr_overall_train_accuracy)
print("Logistic Regression Test Accuracy:", lr_overall_test_accuracy)

# Train Hybrid Model for each label
hybrid_predictions_train = {}  # Store hybrid model predictions on the training set
hybrid_predictions_test = {}   # Store hybrid model predictions on the test set

for label in y_train.columns:
    nb_train_probs = nb_classifiers[label].predict_proba(X_train_dtm)[:, 1]
    lr_train_probs = lr_classifiers[label].predict_proba(X_train_dtm)[:, 1]
    train_probs = (nb_train_probs + lr_train_probs) / 2
    hybrid_pred_train = np.where(train_probs > 0.5, 1, 0)
    hybrid_predictions_train[label] = hybrid_pred_train

    nb_test_probs = nb_classifiers[label].predict_proba(X_test_dtm)[:, 1]
    lr_test_probs = lr_classifiers[label].predict_proba(X_test_dtm)[:, 1]
    test_probs = (nb_test_probs + lr_test_probs) / 2
    hybrid_pred_test = np.where(test_probs > 0.5, 1, 0)
    hybrid_predictions_test[label] = hybrid_pred_test

    train_accuracy = accuracy_score(y_train[label], hybrid_pred_train)
    test_accuracy = accuracy_score(y_test[label], hybrid_pred_test)

    print(f"\nHybrid Model for {label}:")
    print("Training Accuracy (Hybrid Model):", train_accuracy)
    print("Test Accuracy (Hybrid Model):", test_accuracy)
    print("Classification Report (Hybrid Model) for Test Set:")
    print(classification_report(y_test[label], hybrid_pred_test, zero_division=1))

# Calculate overall training and test accuracies for hybrid model
hybrid_train_accuracies = [accuracy_score(y_train[label], hybrid_predictions_train[label]) for label in y_train.columns]
hybrid_test_accuracies = [accuracy_score(y_test[label], hybrid_predictions_test[label]) for label in y_train.columns]

hybrid_overall_train_accuracy = sum(hybrid_train_accuracies) / len(hybrid_train_accuracies)
hybrid_overall_test_accuracy = sum(hybrid_test_accuracies) / len(hybrid_test_accuracies)

print("\n Hybrid Model Training Accuracy:", hybrid_overall_train_accuracy)
print("Hybrid Model Test Accuracy:", hybrid_overall_test_accuracy)

# Transform new input text into vectors
input_text = ['i hate you']
input_text_dtm = vect.transform(input_text)
# Make predictions using Naive Bayes
nb_prediction = {}
for label, nb_classifier in nb_classifiers.items():
    nb_prediction[label] = nb_classifier.predict(input_text_dtm)
# Make predictions using Logistic Regression
lr_prediction = {}
for label, lr_classifier in lr_classifiers.items():
    lr_prediction[label] = lr_classifier.predict(input_text_dtm)
# Hybrid Model Prediction
hybrid_prediction = {}
for label in y_train.columns:
    nb_test_probs = nb_classifiers[label].predict_proba(input_text_dtm)[:, 1]
    lr_test_probs = lr_classifiers[label].predict_proba(input_text_dtm)[:, 1]
    test_probs = (nb_test_probs + lr_test_probs) / 2
    hybrid_pred = np.where(test_probs > 0.5, 1, 0)
    hybrid_prediction[label] = hybrid_pred

# Print Predictions
print("Naive Bayes Prediction:")
for label, prediction in nb_prediction.items():
    print(label + ":", prediction)

print("\nLogistic Regression Prediction:")
for label, prediction in lr_prediction.items():
    print(label + ":", prediction)

print("\nHybrid Model Prediction:")
for label, prediction in hybrid_prediction.items():
    print(label + ":", prediction)

# Bar plot for toxic comments distribution
toxic_counts = train_df.iloc[:, 2:].sum()
plt.figure(figsize=(5, 5))
toxic_counts.plot(kind='bar')
plt.title('Distribution of Toxic Comments')
plt.xlabel('Toxic Category')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.show()

# Plotting comparison graph for all three models
models = ['Naive Bayes', 'Logistic Regression', 'Hybrid']
train_accuracies = [overall_nb_training_accuracy, lr_overall_train_accuracy, hybrid_overall_train_accuracy]
test_accuracies = [overall_nb_test_accuracy, lr_overall_test_accuracy, hybrid_overall_test_accuracy]

# Bar plot for model comparison
models = ['Naive Bayes', 'Logistic Regression', 'Hybrid']
test_accuracies = [overall_nb_test_accuracy, lr_overall_test_accuracy, hybrid_overall_test_accuracy]

plt.figure(figsize=(5, 5))
plt.bar(models, test_accuracies, color=['blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Test Accuracies for Different Models')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrix for Naive Bayes
for label, nb_classifier in nb_classifiers.items():
    y_test_pred_nb = nb_classifier.predict(X_test_dtm)
    plot_confusion_matrix(f'Naive Bayes - {label}', y_test[label], y_test_pred_nb)

# Plot confusion matrix for Logistic Regression
for label, lr_classifier in lr_classifiers.items():
    y_test_pred_lr = lr_classifier.predict(X_test_dtm)
    plot_confusion_matrix(f'Logistic Regression - {label}', y_test[label], y_test_pred_lr)

# Plot confusion matrix for Hybrid Model
for label in y_train.columns:
    y_test_pred_hybrid = hybrid_predictions_test[label]
    plot_confusion_matrix(f'Hybrid Model - {label}', y_test[label], y_test_pred_hybrid)

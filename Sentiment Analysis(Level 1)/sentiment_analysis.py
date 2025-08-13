# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # Fix for latest NLTK
nltk.download('wordnet')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 3. Load dataset
from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# Show columns so you can check names
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


text_col = "clean_text"      # column with tweet text
label_col = "category"       # column with sentiment labels

# 5. Clean text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

url_re = re.compile(r'https?://\S+|www\.\S+')
mention_re = re.compile(r'@\w+')
hashtag_re = re.compile(r'#\w+')
emoji_re = re.compile("["
                      u"\U0001F600-\U0001F64F"
                      u"\U0001F300-\U0001F5FF"
                      u"\U0001F680-\U0001F6FF"
                      u"\U0001F1E0-\U0001F1FF"
                      "]+", flags=re.UNICODE)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = url_re.sub(' ', text)
    text = mention_re.sub(' ', text)
    text = emoji_re.sub(' ', text)
    text = hashtag_re.sub(lambda m: m.group(0)[1:], text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1 and not t.isdigit()]
    return " ".join(tokens)

df['_clean_text'] = df[text_col].astype(str).apply(clean_text)

# 6. Sentiment distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x=label_col, data=df, palette="Set2")
plt.title("Sentiment Distribution")
plt.show()

# 7. Prepare dataset
df = df.dropna(subset=[label_col])
df = df[df['_clean_text'].str.strip() != ""]

le = LabelEncoder()
df['_label'] = le.fit_transform(df[label_col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    df['_clean_text'], df['_label'],
    test_size=0.2, random_state=42, stratify=df['_label']
)

# 8. Pipelines
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)

nb_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', MultinomialNB())
])

svm_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', LinearSVC(max_iter=3000))
])

# 9. Train & evaluate
def evaluate_pipeline(pipeline, X_tr, y_tr, X_te, y_te, name="model"):
    print(f"\nTraining {name} ...")
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_te, preds, target_names=le.classes_))
    cm = confusion_matrix(y_te, preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()
    return pipeline, acc

nb_model, nb_acc = evaluate_pipeline(nb_pipeline, X_train, y_train, X_test, y_test, "Naive Bayes")
svm_model, svm_acc = evaluate_pipeline(svm_pipeline, X_train, y_train, X_test, y_test, "Linear SVM")

# 10. Save best model
best_model = svm_model if svm_acc >= nb_acc else nb_model
joblib.dump({'model_pipeline': best_model, 'label_encoder': le}, "best_sentiment_model.joblib")
files.download("best_sentiment_model.joblib")

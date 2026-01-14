import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from preprocess import clean_text

# Load dataset
df = pd.read_csv("dataset/fake_job_postings.csv")

# Combine text fields
df['text'] = df['title'] + " " + df['description'] + " " + df['requirements']
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['fraudulent']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

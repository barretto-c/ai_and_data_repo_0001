# Sentiment Analysis with NLTK and Scikit-learn
# This script demonstrates how to perform sentiment analysis using the NLTK movie reviews dataset and Scikit-learn.
# It includes data loading, preprocessing, model training, and evaluation.

import nltk
from nltk.corpus import movie_reviews
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Download required NLTK data
#e.g. Files in C:\Users\<YourUsername>\AppData\Roaming\nltk_data\corpora\movie_reviews
nltk.download('movie_reviews')

nltk.download('punkt')

# Load dataset
docs = []
print("Loading movie reviews...")
print(movie_reviews.categories())

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        text = movie_reviews.raw(fileid)
        docs.append((text, category))

print(f"Total documents: {len(docs)}")

texts = [text for text, _ in docs]
# print(f"{len(texts))

labels = [label for _, label in docs]
print(len(labels))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts,   # X: review texts (features)
    labels,  # y: sentiment labels (targets)
    test_size=0.2,
    random_state=42
)

# Define pipeline
# Create a pipeline:
#1 vectorize text,
#2 then classify

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),  # Convert text to TF-IDF features
    ('clf', LogisticRegression(max_iter=1000))       # Train logistic regression classifier
])

# Train model
fitResult = pipeline.fit(X_train, y_train)


# Evaluate
# Use the trained pipeline to predict sentiment labels for the test set
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Example predictions
sample_reviews = [
    "I absolutely loved this movie - ! The plot was thrilling and the characters were well-developed.",
    "This was a terrible film. I wasted two hours of my life watching it.",
    "An average movie with some good moments but overall nothing special.",
    "It was a great movie"
]
predictions = pipeline.predict(sample_reviews)
for review, sentiment in zip(sample_reviews, predictions):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")





#Sentiment Analysis using TextBlob
#TextBlob is a simple library for processing textual data
#Built on top of NLTK 
#Provides a simple API for diving into common natural language processing (NLP) tasks
#Sentiment analysis returns two properties: polarity and subjectivity

#python -m textblob.download_corpora

from textblob import TextBlob

# Sample texts
textBlurbs = [
    "Great, another meeting. Just what I needed.",
    "I’m glad it’s over after a long night.",
    "The food was amazing, but the service was terrible.",
    "He really killed it in the presentation.",
    "To be or not to be, that is the question."
]

# Analyze sentiment
for text in textBlurbs:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    print(f"Text: {text}")
    print(f"Polarity: {polarity:.2f} | Subjectivity: {subjectivity:.2f}")
    print(f"Sentiment: {'Positive' if polarity > 0.05 else 'Negative' if polarity < -0.05 else 'Neutral'}\n")

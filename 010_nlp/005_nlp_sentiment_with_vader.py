#Usesus VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
#VADER is specifically attuned to sentiments expressed in social media

# Scores is a dictionary with keys: 'neg', 'neu', 'pos', and 'compound'
# 'neg', 'neu', and 'pos' represent the proportion of the text that falls into each category
# 'compound' is a normalized score that summarizes the overall sentiment of the text
# Compound score ranges from -1 (most extreme negative) to +1 (most extreme positive)
# Thresholds for sentiment classification:
# Positive sentiment: compound score >= 0.05
# Negative sentiment: compound score <= -0.05
# Neutral sentiment: compound score > -0.05 and < 0.05
# Note: These thresholds can be adjusted based on specific use cases

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required resources (only needed once)
# VADER lexicon
nltk.download('vader_lexicon')

# Initialize varables
sia = SentimentIntensityAnalyzer()

#
textBlurbs = [
    "Great, another meeting. Just what I needed.",
    "I’m glad it’s over after a long night.",
    "The food was amazing, but the service was terrible.",
    "He really killed it in the presentation.",
    "To be or not to be, that is the question."
]

# Analyze sentiment
for text in textBlurbs:
    scores = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Scores: {scores}")
    print(f"Sentiment: {'Positive' if scores['compound'] > 0.05 else 'Negative' if scores['compound'] < -0.05 else 'Neutral'}\n")




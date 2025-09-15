# Title: Predict Candidate Match Score
# This is considered Machine Learning with NLP
# This is NOT Artificial Intelligence with NLP
# calculate candidate match score based on job description and resumes using TF-IDF and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample inputs
job_description = """We are seeking a dedicated and knowledgeable Gardener to maintain and enhance the beauty, health, and functionality of our outdoor spaces. The ideal candidate will have a passion for horticulture, a strong work ethic, and the ability to work independently or as part of a team to ensure our gardens, lawns, and landscapes thrive year-round."""
resume_1 = """I love gardening and have 5 years of experience in landscape maintenance and plant care. """
resume_2 = """I just love eating vegetables"""
resume_3 = """I am a garden manager with over 10 years of experience in horticulture, landscape design, and outdoor space maintenance. I have a strong passion for plants and a proven track record of creating beautiful and sustainable gardens."""

# Combine texts
documents = [job_description, resume_1, resume_2, resume_3]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Compute cosine similarity between job description and each resume
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Output results
for i, score in enumerate(similarities[0], start=1):
    print(f"Resume {i} match score: {score:.2f}")

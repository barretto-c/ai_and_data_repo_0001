import os
import openai
import numpy as np

# Set your OpenAI API key (store securely in environment variables)
# On Windows, you can set the environment variable in Command Prompt before running your script:

# this code lists models available for the given API key

api_key = os.getenv("OPENAI_API_KEY_001")
client = openai.OpenAI(api_key=api_key)

resume_text_1 = "Experienced gardener with deep expertise in vegetable growing."
resume_text_2 = "I am a vice president."
job_description = "Seeking a tomato gardener."

# Get embeddings for both resumes and the job description
resume_vectors = client.embeddings.create(model="text-embedding-3-small", input=[resume_text_1, resume_text_2]).data
job_vector = client.embeddings.create(model="text-embedding-3-small", input=job_description).data[0].embedding

def cosine_similarity(a, b):
	return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

score_1 = cosine_similarity(resume_vectors[0].embedding, job_vector)
score_2 = cosine_similarity(resume_vectors[1].embedding, job_vector)

print(f"Match score for Resume 1: {score_1:.3f}")
print(f"Match score for Resume 2: {score_2:.3f}")
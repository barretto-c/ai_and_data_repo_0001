import openai
import os

# Set your OpenAI API key (store securely in environment variables)
# On Windows, you can set the environment variable in Command Prompt before running your script:

# this code lists models available for the given API key

api_key = os.getenv("OPENAI_API_KEY_001")
client = openai.OpenAI(api_key=api_key)

# Create an embedding for a sample text
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Hello"
)
embedding = response.data[0].embedding
print("Generated Embedding:")
print(embedding)

import openai
import os

# Set your OpenAI API key (store securely in environment variables)
# On Windows, you can set the environment variable in Command Prompt before running your script:

# this code lists models available for the given API key

api_key = os.getenv("OPENAI_API_KEY_001")
client = openai.OpenAI(api_key=api_key)

# List all available models for the current API key
models = client.models.list()
print("Available models:")
for model in models.data:
    print(model.id)
    
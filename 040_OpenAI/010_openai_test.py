import openai
import os

# Set your OpenAI API key (store securely in environment variables)
api_key = os.getenv("OPENAI_API_KEY_001")
client = openai.OpenAI(api_key=api_key)

# Define your messages for the chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

# Call the API using GPT-5 (replace 'gpt-5' with the actual model name if different)
response = client.chat.completions.create(
    model="gpt-5",  # Use the correct model name for GPT-5 if available
    messages=messages,
    temperature=0.7,
    max_completion_tokens=300  # Use 'max_completion_tokens' for GPT-5
)

print("Response from GPT-5:")
print(response.choices[0].message.content)
import openai
import os

# Set your OpenAI API key (store securely in environment variables)
api_key = os.getenv("OPENAI_API_KEY_001")
if not api_key or api_key.strip() in ["", "your_api_key_here", "sk-..."]:
    print("Error: OpenAI API key is missing or appears to be a placeholder.")
    exit(1)
client = openai.OpenAI(api_key=api_key)


def chat_with_openai(user_input):
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo" if you prefer
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

# Loop for interaction
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("Bot:", chat_with_openai(user_input))
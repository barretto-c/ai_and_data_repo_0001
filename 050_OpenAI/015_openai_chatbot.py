import openai
import os

# Set your OpenAI API key (store securely in environment variables)
api_key = os.getenv("OPENAI_API_KEY_001")
if not api_key or api_key.strip() in ["", "your_api_key_here", "sk-..."]:
    print("Error: OpenAI API key is missing or appears to be a placeholder.")
    exit(1)
client = openai.OpenAI(api_key=api_key)

# Initialize conversation history with system prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def chat_with_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo" if you prefer
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content

# Loop for interaction
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    # Get assistant response
    assistant_reply = chat_with_openai(messages)
    print("Bot:", assistant_reply)
    # Add assistant message to history
    messages.append({"role": "assistant", "content": assistant_reply})
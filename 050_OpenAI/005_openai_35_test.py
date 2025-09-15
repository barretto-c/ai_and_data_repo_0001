import openai
import os

api_key = os.getenv("OPENAI_API_KEY_001")
client = openai.OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",    
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print("Response from GPT")
print(response.choices[0].message.content)

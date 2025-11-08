from openai import AzureOpenAI

# Create a client for Azure OpenAI using your Azure subscription details
api_key = "your_azure_openai_key_here"
api_version = "2023-12-01-preview"
azure_endpoint = "https://your-azure-openai-resource.openai.azure.com/"

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint  # Your Azure OpenAI resource endpoint
)

# Send a chat completion request to your Azure OpenAI deployment
# Note: 'model' here is your Azure deployment name, not the model name
response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        # System prompt: sets the behavior, role, or instructions for the AI assistant
        {"role": "system", "content": "You are a helpful gardening assistant."},
        # User message: actual input or query from the user
        {"role": "user", "content": "What is the best fertizizer for tomatoes in the spring on the east coast inn northern virginia."}
    ]
)

# Print the assistant's response to the console

choice_number = 1
for choice in response.choices:
	print(f"Choice #{choice_number}  \n{choice.message.content}")
	choice_number += 1
	




print(response.choices[0].message.content)

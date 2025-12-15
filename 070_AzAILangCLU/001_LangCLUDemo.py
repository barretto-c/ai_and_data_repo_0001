from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
key = os.getenv("AZURE_LANGUAGE_KEY")
project = os.getenv("AZURE_CLU_PROJECT")
deployment = os.getenv("AZURE_CLU_DEPLOYMENT", "production")

if not all([endpoint, key, project, deployment]):
    raise ValueError("Please set AZURE_LANGUAGE_ENDPOINT, AZURE_LANGUAGE_KEY, AZURE_CLU_PROJECT, and AZURE_CLU_DEPLOYMENT in your .env file.")

client = ConversationAnalysisClient(endpoint, AzureKeyCredential(key))

with client:
    result = client.analyze_conversation(
        task={
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "participantId": "user",
                    "text": "Order me a small pizza with extra cheese and a coke"
                }
            },
            "parameters": {
                "projectName": project,
                "deploymentName": deployment
            }
        }
    )


# --- Enhanced Output for SDK Response ---
def print_clu_response(response, label="SDK"):
    prediction = response["result"]["prediction"]
    print(f"\n[{label} RESPONSE]")
    print("Query:", response["result"].get("query", "<not available>"))
    print("Top Intent:", prediction["topIntent"])
    print("Intents:")
    for intent in prediction.get("intents", []):
        print(f" - {intent['category']}: {intent['confidenceScore']}")
    print("Entities:")
    for entity in prediction.get("entities", []):
        print(f" - {entity['category']}: {entity['text']} (confidence: {entity.get('confidenceScore')})")
        if "resolutions" in entity:
            print("   Resolutions:")
            for res in entity["resolutions"]:
                print(f"     - {res}")
        if "extraInformation" in entity:
            print("   Extra Information:")
            for extra in entity["extraInformation"]:
                print(f"     - {extra}")

print_clu_response(result, label="SDK")

# --- REST API Example (if present) ---
try:
    rest_result
except NameError:
    rest_result = None
if rest_result:
    print_clu_response(rest_result, label="REST API")

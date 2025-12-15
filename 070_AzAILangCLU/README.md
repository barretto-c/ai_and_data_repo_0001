
# 001_LangCLUDemo.py Setup and Prerequisites

This project demonstrates how to use Azure Cognitive Services (Conversational Language Understanding, CLU) with Python. Follow these steps to set up your environment and Azure resources before running the demo script.

## 1. Python Environment Setup

1. **Install Python 3.8 or higher**  
   Download from [python.org](https://www.python.org/downloads/).

2. **(Recommended) Create a Virtual Environment**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Upgrade pip**
   ```sh
   python -m pip install --upgrade pip
   ```

4. **Install Required Python Packages**
   ```sh
   pip install -r requirements.txt
   ```

## 2. Azure Cognitive Services Setup

> **Note:** As of this writing, the US West region is recommended for CLU. US East may not have full support for all CLU features.

1. **Create a Resource Group (if needed)**
   ```sh
   az group create --name <your-resource-group> --location westus2
   ```

2. **Create a Multi-Service Cognitive Services Account (with Language enabled)**
   ```sh
   az cognitiveservices account create --name <your-account-name> --resource-group <your-resource-group> --kind TextAnalytics --sku F0 --location westus2 --yes
   ```

3. **Get Your Endpoint and Key**
   ```sh
   az cognitiveservices account show --name <your-account-name> --resource-group <your-resource-group> --query properties.endpoint -o tsv
   az cognitiveservices account keys list --name <your-account-name> --resource-group <your-resource-group> --query key1 -o tsv
   ```

## 3. Create and Configure a CLU Project

1. Go to [Azure Language Studio](https://language.azure.com/).
2. Select your subscription and the Cognitive Services account you created.
3. Create a new CLU project:
   - Define your schema (intents and entities).
   - Example intents: `OrderPizza`, `CancelOrder` (add sample utterances for each intent).
   - Example entities: `PizzaType`, `Topping`, `Size`, `Quantity` (define as lists, e.g., cheese, pepperoni, small, large).
4. Train and deploy your project (e.g., to "production").

## 4. Train, Review, and Deploy Your CLU Model

1. In Azure Language Studio:
   - Create your training jobs and train your model.
   - Review your model's performance metrics.
   - Deploy your model to a deployment slot (e.g., "production").
   - Test your deployment in the portal to ensure it works as expected.

## 5. Configure Your .env File

1. After your Azure resource and CLU project are ready, copy the provided `.env.sample` file to `.env`:
   ```sh
   cp .env.sample .env  # On Windows: copy .env.sample .env
   ```
2. Open `.env` and fill in your actual Azure endpoint, key, CLU project name, and deployment name.

Example:
```
AZURE_LANGUAGE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=your-key-here
AZURE_CLU_PROJECT=your-clu-project-name
AZURE_CLU_DEPLOYMENT=production
```

## 6. Run the Demo Script

```sh
python 001_LangCLUDemo.py
```

```
Your output should look like this
[SDK RESPONSE]
Query: Order me a small pizza with extra cheese and a coke
Top Intent: OrderPizza
Intents:
 - OrderPizza: 0.7955833
 - CancelOrder: 0.68590707
 - None: 0
Entities:
 - Size: small (confidence: 1)
   Extra Information:
     - {'extraInformationKind': 'ListKey', 'key': 'small'}
 - PizzaType: cheese (confidence: 1)
   Extra Information:
     - {'extraInformationKind': 'ListKey', 'key': 'cheese'}
```
---

For more information, see the [Azure Language documentation](https://learn.microsoft.com/azure/ai-services/language-service/).



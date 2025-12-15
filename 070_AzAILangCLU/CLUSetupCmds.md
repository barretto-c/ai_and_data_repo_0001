az cognitiveservices account create --name 2026CertTextAna01 --resource-group 2026CertAccountWest2 --kind TextAnalytics --sku S --location westus2 --yes


az cognitiveservices account show --name 2026CertTextAna01 --resource-group 2026CertAccountWest2 --query "kind" "TextAnalytics"

az rest --method put --uri "https://westus2.api.cognitive.microsoft.com/language/authoring/analyze-conversations/projects/2026CLUWest2Project01?api-version=2023-04-01" --headers "Ocp-Apim-Subscription-Key=9fef6e43e2224fa8ae223f16950086ec" --body '{"projectKind":"Conversation","language":"en-us"}'

Create CLU Project under west2

az rest --method get --uri "https://https://westus2.api.cognitive.microsoft.com//language/authoring/analyze-conversations/projects?api-version=2023-04-01" --headers "Ocp-Apim-Subscription-Key=9fef6e43e2224fa8ae223f16950086ec"

az rest --method get --uri "https://eastus.api.cognitive.microsoft.com/language/authoring/analyze-conversations/projects?api-version=2023-04-01" --headers "Ocp-Apim-Subscription-Key=226f387cb8904c84a321c10955dfc704"


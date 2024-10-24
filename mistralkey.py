import requests
import json

# Define the API endpoint and your API key
API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "t7kTZjaRyxSJ4fHUp8xILpKG0FXlijE9"

# Define the headers with your API key
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Define the data for the API request
data = {
    "model": "mistral-small",
    "messages": [
        {"role": "user", "content": "Say this is a test"}
    ],
}

# Make the API request
response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

# Check the response status code
if response.status_code == 200:
    print("API key is working. Here's the response:")
    print(response.json())
else:
    print(f"API key is not working. Error code: {response.status_code}")
    print(response.text)

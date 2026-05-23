import requests
import json

API_KEY = "YOUR_API_KEY"

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
user_prompt = input("Ask something:")
data = {
    "model": "openai/gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": user_prompt}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json()["choices"][0]["message"]["content"])

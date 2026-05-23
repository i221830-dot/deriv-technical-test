from openai import OpenAI
import os

API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(api_key = API_KEY)

response = client.chat.completions.create( model = "gpt-4o-mini",messages = [{"role":"user","content":"Hello"}])

print(response.choices[0].message.content)

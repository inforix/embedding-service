import requests

url = "http://127.0.0.1:8000/v1/embeddings"
data = {
    "model": "tao-8k",  # You can change this if needed
    "input": ["This is a test sentence.", "Embedding compatible with OpenAI."]
}

response = requests.post(url, json=data)
print(response.json())
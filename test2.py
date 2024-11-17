import requests

url = "http://10.81.247.93/v1/embeddings"
data = {
    "model": "tao-8k",  # You can change this if needed
    "input": ["This is a test sentence.", "Embedding compatible with OpenAI."]
}

response = requests.post(url, json=data)
print(response.json())
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load the TAO-8K model and tokenizer (make sure the model is available on Hugging Face or local path)
# Replace 'tao-8k-model-name' with the actual model identifier or path to your model
tokenizer = AutoTokenizer.from_pretrained("tao-8k")
model = AutoModel.from_pretrained("tao-8k")

# Initialize FastAPI app
app = FastAPI()

# Define the request model for input text
class EmbeddingRequest(BaseModel):
    model: str  # Model name (to match OpenAI API structure)
    input: list  # List of input texts to be embedded

# Define the response model
class EmbeddingResponse(BaseModel):
    data: list  # List of embedding objects
    object: str = "list"  # This mimics OpenAI's 'object' field
    model: str = "tao-8k"  # Adjusted to reflect the TAO-8K model

@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    if request.model != "tao-8k":
        return {"error": "Unsupported model"}  # Ensure the requested model is 'tao-8k-model'

    # Generate embeddings for each input text
    embeddings = []
    for text in request.input:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            # Generate the embedding (this assumes your model has a last_hidden_state output)
            embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        embeddings.append({"embedding": embedding, "index": len(embeddings)})

    # Return the OpenAI-like response
    return EmbeddingResponse(data=embeddings, model=request.model)

import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

_ = load_dotenv()

# Load the TAO-8K model and tokenizer (make sure the model is available on Hugging Face or local path)
# Replace 'tao-8k-model-name' with the actual model identifier or path to your model
#tokenizer = AutoTokenizer.from_pretrained(os.environ.get('MODEL_PATH'))
model = SentenceTransformer(os.environ.get('MODEL_PATH'))

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
    model: str = os.environ.get('MODEL_NAME')  # Adjusted to reflect the TAO-8K model



@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    if request.model != "tao-8k":
        return {"error": "Unsupported model"}  # Ensure the requested model is 'tao-8k-model'

    # Generate embeddings for each input text
    
    embeddings = model.encode(request.input, normalize_embeddings=True)
    # for text in request.input:
    #     print(type(text))
    #     # Tokenize the input text
    #     embedding = model.encode(text, normalize_embeddings=True)
    #     embeddings.append({"embedding": embedding.tolist(), "index": len(embeddings)})
    ret = []
    for i, embedding in enumerate(embeddings):
        print(f'i: {i}')
        obj = {
            "object": "embedding",
            "index": i,
            "embedding": embedding.tolist()
        }
        ret.append(obj)
    

    # Return the OpenAI-like response
    return EmbeddingResponse(data=ret, model=request.model)

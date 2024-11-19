import os
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pathlib
import tiktoken

_ = load_dotenv()

tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
model_paths = {}
model_base_path = os.getenv('MODEL_BASE_PATH')
# Load the TAO-8K model and tokenizer (make sure the model is available on Hugging Face or local path)
# Replace 'tao-8k-model-name' with the actual model identifier or path to your model
#tokenizer = AutoTokenizer.from_pretrained(os.environ.get('MODEL_PATH'))
def load_model(model_name: str):
    model = model_paths.get(model_name)
    if model is None:
        path = os.path.join(model_base_path, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path {path} does not exist.")
        model = SentenceTransformer(path)
        model_paths[model_name] = model
        
        if model_name == 'gte_Qwen2-7B-instruct':
            model.max_seq_length = 8192
    
    return model

# Initialize FastAPI app
app = FastAPI()

# Define the request model for input text
class EmbeddingRequest(BaseModel):
    model: str  # Model name (to match OpenAI API structure)
    input: List[Union[str, List[int]]]  # Accept raw text or tokenized integers

# Define the response model
class EmbeddingResponse(BaseModel):
    data: list  # List of embedding objects
    object: str = "list"  # This mimics OpenAI's 'object' field
    model: str = os.environ.get('MODEL_NAME')  # Adjusted to reflect the TAO-8K model



@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    try:
        model = load_model(request.model)
        
        texts = request.input
        if isinstance(request.input, List[List[int]]):
            """Tokenized """
            texts = [tiktoken_encoding.decode(x) for x in request.input]
        
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        ret = [ {
                "object": "embedding",
                "index": idx,
                "embedding": emb
            } for idx, emb in enumerate(embeddings)]
            
        # Return the OpenAI-like response
        return EmbeddingResponse(data=ret, model=request.model)
    except Exception as e:
        return {"error": str(e)}

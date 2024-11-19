import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pathlib

_ = load_dotenv()

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
    input: list  # List of input texts to be embedded

# Define the response model
class EmbeddingResponse(BaseModel):
    data: list  # List of embedding objects
    object: str = "list"  # This mimics OpenAI's 'object' field
    model: str = os.environ.get('MODEL_NAME')  # Adjusted to reflect the TAO-8K model



@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    try:
        model = load_model(request.model)
        ret = []
        for i, text in enumerate(request.input):
            print(text[:50])
            # Generate embeddings for each input text
            embeddings = model.encode([text], normalize_embeddings=True)
        
            # print(f'i: {i}')
            obj = {
                "object": "embedding",
                "index": i,
                "embedding": embeddings[0].tolist()
            }
            print(obj.embedding)
            ret.append(obj)

        # Return the OpenAI-like response
        return EmbeddingResponse(data=ret, model=request.model)
    except Exception as e:
        return {"error": str(e)}

from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer, util
import json

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FastAPI
app = FastAPI()

# Fungsi untuk memuat kata kunci filter
def load_filter_keywords():
    with open("filter_keywords.json", "r") as file:
        return json.load(file)["keywords"]

# Fungsi untuk memeriksa keamanan prompt menggunakan embeddings
def is_prompt_safe_with_embeddings(prompt, keywords, threshold=0.5):
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    similarities = util.cos_sim(prompt_embedding, keyword_embeddings)

    for sim in similarities[0]:
        if sim > threshold:
            return False
    return True

# Fungsi untuk memeriksa kata kunci eksplisit
def contains_keywords(prompt, keywords):
    return any(keyword.lower() in prompt.lower() for keyword in keywords)

# Fungsi utama untuk memeriksa keamanan prompt
@app.post("/check_prompt")
async def check_prompt(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        keywords = load_filter_keywords()
        is_safe_embeddings = is_prompt_safe_with_embeddings(prompt, keywords)
        contains_explicit_keywords = contains_keywords(prompt, keywords)

        safe = is_safe_embeddings and not contains_explicit_keywords
        return {"safe": safe}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

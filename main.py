import requests
import logging
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

connections.connect("default", host="localhost", port="19530")

model = SentenceTransformer('all-MiniLM-L6-v2')

hf_token = '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'

client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token=hf_token,
)

class WikipediaURL(BaseModel):
    url: str

class Query(BaseModel):
    query: str

def extract_wikipedia_content(url: str) -> list:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch Wikipedia page")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find(id="mw-content-text")
    
    if not content:
        raise HTTPException(status_code=400, detail="Failed to extract content from Wikipedia page")

    paragraphs = content.find_all('p')
    return [p.get_text().strip() for p in paragraphs if p.get_text().strip()]

def create_milvus_collection(collection_name: str):
    try:
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists. Dropping it.")
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, "Wikipedia content")
        collection = Collection(collection_name, schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        logger.info(f"Collection '{collection_name}' created successfully.")
        return collection
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise

@app.post("/load")
async def load_data(wiki_url: WikipediaURL):
    try:
        logger.info(f"Attempting to load data from URL: {wiki_url.url}")
        paragraphs = extract_wikipedia_content(wiki_url.url)
        logger.info(f"Extracted {len(paragraphs)} paragraphs")

        logger.info("Creating embeddings")
        embeddings = model.encode(paragraphs)

        logger.info("Creating/getting Milvus collection")
        collection = create_milvus_collection("wikipedia_content")

        logger.info("Inserting data into Milvus")
        entities = [
            [p for p in paragraphs],
            embeddings.tolist()
        ]
        collection.insert(entities)
        collection.flush()
        
        logger.info("Data loaded successfully")
        return {"message": "Data loaded successfully", "paragraphs_count": len(paragraphs)}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(query_input: Query):
    logger.info(f"Received query: {query_input.query}")
    try:
        # Connect to the Milvus collection
        collection = Collection("wikipedia_content")
        collection.load()

        # Create an embedding for the query
        query_embedding = model.encode([query_input.query])[0].tolist()

        # Search in Milvus
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding], 
            anns_field="embedding", 
            param=search_params,
            limit=5,
            output_fields=["content"]
        )

        # Extract the results
        similar_paragraphs = [hit.entity.get('content') for hit in results[0]]

        # Generate an answer using Hugging Face Inference API
        prompt = f"Based on the following paragraphs, answer the question: {query_input.query}\n\n"
        for paragraph in similar_paragraphs:
            prompt += f"{paragraph}\n\n"
        
        messages = [{"role": "user", "content": prompt}]
        response_text = ""
        try:
            for message in client.chat_completion(
                messages=messages,
                max_tokens=150,
                stream=True,
            ):
                response_text += message.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error during chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail="Error during chat completion")

        answer = response_text.strip()

        return {"answer": answer, "similar_paragraphs": similar_paragraphs}
    except Exception as e:
        logger.error(f"An error occurred during query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Wikipedia Content API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
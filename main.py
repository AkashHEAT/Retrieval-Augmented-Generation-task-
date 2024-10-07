import requests
import logging
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from huggingface_hub import InferenceClient
from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, utility
from sentence_transformers import util 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

connections.connect("default", host="localhost", port="19530")

model_name = 'all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

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

def create_milvus_collection(collection_name: str, documents: list):
    try:

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        vectorstore = Milvus.from_texts(
            texts=documents,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={"host": "localhost", "port": "19530"},
        )
        
        logger.info(f"Collection '{collection_name}' created successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise

@app.post("/load")
async def load_data(wiki_url: WikipediaURL):
    try:
        logger.info(f"Attempting to load data from URL: {wiki_url.url}")
        paragraphs = extract_wikipedia_content(wiki_url.url)
        logger.info(f"Extracted {len(paragraphs)} paragraphs")

        logger.info("Creating/getting Milvus collection")
        vectorstore = create_milvus_collection("wikipedia_content", paragraphs)

        logger.info("Data loaded successfully")
        return {"message": "Data loaded successfully", "paragraphs_count": len(paragraphs)}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(query_input: Query):
    logger.info(f"Received query: {query_input.query}")
    try:
        # Connect to the Milvus collection using Langchain
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name="wikipedia_content",
            connection_args={"host": "localhost", "port": "19530"}
        )

        results = vectorstore.similarity_search(query_input.query, k=5)


        similar_paragraphs = [doc.page_content for doc in results]

        if not similar_paragraphs:
            return {"answer": "I don't have knowledge on the question you asked about.", "similar_paragraphs": []}

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

        query_embedding = embeddings.embed_query(query_input.query)
        paragraph_embeddings = embeddings.embed_documents(similar_paragraphs)

        similarities = util.cos_sim(query_embedding, paragraph_embeddings)

        threshold = 0.3 
        if similarities.max() < threshold:
            return {"answer": "I don't have knowledge on the question you asked about.", "similar_paragraphs": similar_paragraphs}

        return {"answer": answer, "similar_paragraphs": similar_paragraphs}
    except Exception as e:
        logger.error(f"An error occurred during query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Wikipedia Content API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
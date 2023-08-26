import chromadb
from dotenv import load_dotenv
import os
from loguru import logger
import openai
from typing import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import constants as cts

class Chromadb:
    def __init__(self, query):
        # set user query
        self.query = query
        # initialize chroma client
        self.chroma_client = chromadb.HttpClient(host=cts.CHROMADB_EC2_HOST, port=8000)
        # get chroma collection
        self.chromadb_collection = self.chroma_client.get_or_create_collection(name=cts.CHROMADB_COLLECTION_NAME)
        # get openai api key
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    def get_nearest_chunks(self):
        # set openai api key
        openai.api_key = self.openai_api_key
        # get embedding for the user query
        query_embedding = self.get_embedding(self.query)
        # get top results
        results = self.chromadb_collection.query(query_embeddings=query_embedding, n_results=10)
        return results['ids'][0]
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> List[float]:
        response: List[float] = openai.Embedding.create(
            input=text, model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    
# if __name__ == "__main__":
#     query = "what is god?"
    
#     chromadb = Chromadb(query)
#     chromadb.get_nearest_chunks()
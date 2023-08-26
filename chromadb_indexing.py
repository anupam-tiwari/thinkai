import chromadb
import os
import json
from loguru import logger
from fire import Fire
from typing import Optional

from openai_embeddings import get_embedding

chromadb_collection_name = "philosophy"
ec2_host = "54.144.129.4"


def get_local_chroma_client(chromadb_folder: str):
    chroma_client = chromadb.PersistentClient(path=chromadb_folder)
    return chroma_client


def get_remote_chroma_client():
    chroma_client = chromadb.HttpClient(host=ec2_host, port=8000)
    return chroma_client


def get_chromadb_collection(chromadb_folder: Optional[str] = None):
    chroma_client = get_remote_chroma_client()
    collection = chroma_client.get_or_create_collection(name=chromadb_collection_name)
    # if collection.count() > 0:
    #     logger.info(f"deleting collection: {chromadb_collection_name}")
    #     chroma_client.delete_collection(chromadb_collection_name)
    return chroma_client, collection


def index_embeddings(embedding_dirpath: str, collection: chromadb.Collection):
    logger.info(f"embedding dirpath: {embedding_dirpath}")
    logger.info(
        f"number of documents in embeddings dir: {len(os.listdir(embedding_dirpath))}"
    )
    for filename in os.listdir(embedding_dirpath):
        if filename.endswith(".json"):
            filepath = os.path.join(embedding_dirpath, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            for filename, value in data.items():
                url, embedding = value["url"], value["embedding"]
                collection.add(
                    embeddings=embedding,
                    ids=filename,
                    documents=filename,
                    metadatas={"url": url},
                )


def index(embedding_path: str, chromadb_path: str) -> chromadb.Collection:
    chroma_client, collection = get_chromadb_collection(chromadb_folder=chromadb_path)
    index_embeddings(
        embedding_dirpath=embedding_path,
        collection=collection,
    )
    logger.info(f"number of documents in collection: {collection.count()}")
    return collection


def query(collection: chromadb.Collection, query_string: str):
    query_embedding = get_embedding(query_string)
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    logger.info(f"results: {results}")


def main(dirpath: str):
    embedding_path = os.path.join(dirpath, "embeddings")
    chromadb_path = os.path.join(dirpath, "chromadb")
    # collection = index(embedding_path=embedding_path, chromadb_path=chromadb_path)
    chroma_client, collection = get_chromadb_collection(chromadb_folder=chromadb_path)
    query(collection=collection, query_string="What is the meaning of life?")


if __name__ == "__main__":
    Fire(main)


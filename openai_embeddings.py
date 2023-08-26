import openai
from dotenv import load_dotenv
import os
from typing import List, Dict
from fire import Fire
import json
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text: str) -> List[float]:
    response: List[float] = openai.Embedding.create(
        input=text, model="text-embedding-ada-002"
    )
    embedding = response["data"][0]["embedding"]
    return embedding


def get_text_files(dirpath: str) -> Dict[str, str]:
    processed_files = []
    template_dict = {"filename": "", "text": "", "url": ""}

    for filename in os.listdir(dirpath):
        if filename.endswith(".json"):
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            num_chunks = data["chunks_count"]
            chunked_data: List[str] = data["chunks"]
            url = data["url"]
            for i in range(num_chunks):
                new_dict = template_dict.copy()
                new_dict["filename"] = f"{filename}_{i}"
                new_dict["text"] = chunked_data[i]
                new_dict["url"] = url
                processed_files.append(new_dict)
    return processed_files


def main(text_dirpath: str):
    # read all text files in dirpath
    processed_files: List[Dict[str, str]] = get_text_files(text_dirpath)
    logger.info(f"number of text files: {len(processed_files)}")

    # make a new directory for storing embeddings if it doesn't exist
    parent_dir = os.path.dirname(text_dirpath)
    embeddings_dir = os.path.join(parent_dir, "embeddings")
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # get embeddings for each text file
    embeddings_dict = {}

    for i, file_dict in enumerate(processed_files):
        filename, text, url = file_dict["filename"], file_dict["text"], file_dict["url"]
        embedding: List[float] = get_embedding(text)
        embeddings_dict[filename] = {
            "embedding": embedding,
            "url": url,
            "filename": filename,
        }

        # flush to disk if we have 100 embeddings
        if (i + 1) % 100 == 0:
            with open(os.path.join(embeddings_dir, f"{(i+1)}.json"), "w") as f:
                json.dump(embeddings_dict, f)
                embeddings_dict = {}
                logger.info(f"Saved {(i+1)} embeddings to disk")

    # flush the remaining embeddings to disk
    with open(os.path.join(embeddings_dir, f"last_chunk.json"), "w") as f:
        json.dump(embeddings_dict, f)
        logger.info(f"Saved all embeddings to disk")


if __name__ == "__main__":
    Fire(main)


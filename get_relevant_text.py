# import libraries
from get_nearest_chunks import Chromadb
import json
import os


class GetRelevantText:
    def __init__(self, query: str = None):
        self.query = query

    def get_relevant_text(self) -> str:
        # get nearest chunks from chromadb
        chromadb = Chromadb(self.query)
        chunk_links = chromadb.get_nearest_chunks()

        chunk_texts = self.get_chunk_texts(chunk_links)
        final_text_extract = " ".join(chunk_texts)

        return final_text_extract

    def get_chunk_texts(self, chunk_links: list):
        chunk_texts = []

        for link in chunk_links:
            # get file name, chunk no from link
            file_name = link.split("json_")[0] + "json"
            chunk_no = int(link.split("json_")[1])
            # get json file path
            json_file_path = os.path.join("./chunked/data", file_name)
            # read json file
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_dict = json.load(f)
            # get chunk
            chunk = json_dict["chunks"][chunk_no]
            chunk_texts.append(chunk)

        return chunk_texts


# if __name__ == "__main__":
#     query = "what is god?"
#     grt = GetRelevantText(query)
#     print(grt.get_relevant_text())

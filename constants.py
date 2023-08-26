# define constants

# get response
SUMMARIES_FILE_PATH = 'summaries.json'
OPENAI_GPT_MODEL = 'gpt-3.5-turbo'

# chromadb
CHROMADB_DATA_ZIP_FILE_PATH = 'data.zip'
CHROMADB_PERSISTANT_DATA_FOLDER = 'data/'
CHROMADB_COLLECTION_NAME = 'sep-blogs'
CHROMADB_EMBEDDING_MODEL = 'hkunlp/instructor-base'

# articles files path
FILES_LIST_FILE_PATH = './sorted_json_files.json'

# tokenizer
ENCODING_MODEL = 'text-embedding-ada-002'

# chunk max size
CHUNK_MAX_SIZE = 8000

# chunked files path
CHUNKED_JSON_FOLDER = './chunked/'

# chroma db
CHROMADB_COLLECTION_NAME = 'philosophy'
CHROMADB_EC2_HOST = "54.144.129.4"
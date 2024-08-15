from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import sys
import os
import pickle
from dotenv import load_dotenv

# Load env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
print("env variables successfully loaded")

# setting up the model
llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="bystander-rag-model",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="bystander-embedding-model",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

Settings.llm = llm
Settings.embed_model = embed_model
print("Models successfully loaded")

# index the document
print("starting document index")
documents = SimpleDirectoryReader(
    input_files=["/Users/tharun/Desktop/Programming Things/AI_Model_2/combined.txt"]
).load_data()
index = VectorStoreIndex.from_documents(documents)
print("successfully indexed document.")

# save it offline
with open('bystander_index.pkl', 'wb') as f:
    pickle.dump(index, f)
print("succesfully saved offline")

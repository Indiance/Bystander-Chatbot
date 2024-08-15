import os
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import pickle
from llama_index.core import Settings
import streamlit as st
import time

# Load env variables
@st.cache_resource
def load_env_vars():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("OPENAI_API_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    print("env variables successfully loaded")
    return api_key, azure_endpoint, api_version

api_key, azure_endpoint, api_version = load_env_vars()

# Initialize LLM and embedding models
@st.cache_resource
def initialize_models(api_key, azure_endpoint, api_version):
    llm = AzureOpenAI(
        model="gpt-35-turbo-16k",
        deployment_name="bystander-rag-model",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name="bystander-embedding-model",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

llm, embed_model = initialize_models(api_key, azure_endpoint, api_version)

# Load pickle file
@st.cache_resource
def load_index():
    with open('bystander_index.pkl', 'rb') as file:
        index = pickle.load(file)
    print("Pickle successfully loaded")
    query_engine = index.as_query_engine()
    return query_engine

query_engine = load_index()

st.header("Chat with the Bystander 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the Bystander!"}
    ]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t_0 = time.time()
            answer = query_engine.query(st.session_state.messages[-1]["content"])
            print(f'Time taken for that is {time.time() - t_0}')
            st.write(answer.response)
            message = {"role": "assistant", "content": answer.response}
            st.session_state.messages.append(message) # Add response to message history
